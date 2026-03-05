#!/usr/bin/env python3
"""
GRPO Training Script for Recommendation System
TRL의 GRPOTrainer를 사용한 추천 시스템 학습
RetrievalService와 연동하여 NDCG 기반 리워드로 학습
"""

import os
import ray
import torch
import argparse
import json
import gc
import numpy as np
import logging as std_logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import contextlib

from trl import GRPOTrainer, GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from utils.reward_function import (
    RecRewardFrunction, 
    calculate_ndcg, 
    calculate_hit_rate,
    LocalEmbeddingRewardFunction,
    SimilarHistoryItemMentionReward,
    MetadataMentionReward,
    FormatComplianceReward,
    ItemPreferenceMentionReward,
)
from utils.dataset import create_dataloaders
from evaluator import RecommendationEvaluator

from accelerate import logging
from accelerate.utils import gather

import torch.distributed as dist
from numba import cuda

import wandb
import random

logger = logging.get_logger(__name__)

class GRPOTrainerRecReward(GRPOTrainer):

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations

        # print(f"inputs: {inputs[0].keys()}")
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # print(f"reward_kwargs: {reward_kwargs}")
        # print(f"keys: {reward_kwargs.keys()}")

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # Reward breakdown 수집을 위한 딕셔너리
        reward_breakdowns = {}

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    generated_texts=completions,
                    targets=reward_kwargs["target"],
                    histories=reward_kwargs["history"],
                    user_ids=reward_kwargs["user_id"],
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                
                # Reward breakdown 정보 수집 (LocalEmbeddingRewardFunction인 경우)
                if reward_func_name == "LocalEmbeddingRewardFunction":
                    breakdown = reward_func.get_reward_breakdown()
                    for key, value in breakdown.items():
                        if key not in reward_breakdowns:
                            reward_breakdowns[key] = []
                        reward_breakdowns[key].append(value)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        
        # --- 변경된 부분: 직접 로깅 대신 인스턴스 변수에 저장 ---
        # 메인 프로세스에서만 계산하여 저장
        if reward_breakdowns and self.accelerator.is_main_process:
            self._store_reward_breakdown(reward_breakdowns)
        
        return rewards_per_func
    
    def _store_reward_breakdown(self, reward_breakdowns):
        """
        계산된 Breakdown 통계치를 self._stored_metrics에 저장합니다.
        실제 로깅은 log() 메서드가 호출될 때 수행됩니다.
        """
        mode = "train" if self.model.training else "eval"
        for key, values_list in reward_breakdowns.items():
            # 여러 reward function에서 수집된 값들을 합침
            all_values = torch.cat(values_list) if len(values_list) > 0 else torch.tensor([])
            if len(all_values) > 0:
                self._metrics[mode][f"rewards/{key}"] = all_values.tolist()
                # self._metrics[mode][f"rewards/{key}_std"] = all_values

class GRPOTrainerWrapper:
    """
    TRL GRPO를 활용한 추천 시스템 학습기
    """
    
    def __init__(self, args):
        self.args = args

        
        # Ray 초기화 (use_local_embedding이 False인 경우에만)
        if not args.use_local_embedding:
            if not ray.is_initialized():
                print(f"🔧 Initializing Ray...")
                ray.init(address=args.ray_address, namespace=args.namespace)
                print(f"✓ Ray initialized")
        
        # 토크나이저 로드
        print(f"📚 Loading tokenizer: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        print(f"🤖 Loading model: {args.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        

        
        # 데이터로더 생성 (create_dataloaders 함수 사용)
        (
            self.train_dataset,
            self.valid_dataset,
            self.test_dataset,
            self.prompt_generator,
            self.item_metadata,
        ) = create_dataloaders(args, tokenizer=self.tokenizer)
        
        if args.num_epochs > 0:
                    
            # wandb 초기화 및 args 전달
            if args.report_to == "wandb":
                print(f"📊 Initializing Weights & Biases...")
                wandb.init(
                    project="rl4rec",
                    name=args.run_name,
                    config=vars(args),  # args의 모든 요소를 wandb config로 전달
                )
                print(f"✓ Wandb initialized with all args")

            # GRPO Config
            # vLLM 모드 결정: 서버 모드 또는 colocate 모드
            vllm_mode = "server" if args.use_vllm_server else "colocate"
            vllm_max_model_length=args.max_length+args.max_new_tokens
            
            print(f"🔧 vLLM Mode: {vllm_mode}")
            if args.use_vllm_server:
                print(f"   vLLM Server URL: {args.vllm_server_url}")
                print(f"   Training GPU ID: {args.train_gpu_id}")
                args.vllm_gpu_memory_utilization = None
                args.vllm_max_model_length = None
                args.vllm_enable_sleep_mode = None
                args.vllm_importance_sampling_correction = None
                args.vllm_importance_sampling_mode = None
                args.vllm_importance_sampling_cap = None
                args.vllm_max_model_length = None
            
            grpo_config = GRPOConfig(
                output_dir=args.checkpoint_dir,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,

                logging_steps=args.log_interval,
                eval_steps=args.eval_interval,
                save_steps=args.save_interval,
                save_total_limit=args.save_total_limit,
                dataloader_num_workers=16,
                
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                bf16=True,
                report_to=args.report_to if args.report_to != "none" else None,
                run_name=args.run_name,
                beta=args.reference_model_kld_coef,
                
                # Loss type
                loss_type=args.loss_type,
                importance_sampling_level=args.importance_sampling_level,
                
                # GRPO specific
                num_generations=args.num_sample_generations,
                temperature=args.train_temperature,
                max_steps=args.max_steps,

                # Generation
                max_completion_length=args.max_new_tokens,
                repetition_penalty=1.0,
                top_p=0.95,

                # Save
                # load_best_model_at_end=True,
                # metric_for_best_model="reward",
                # greater_is_better=True,
                # eval_strategy="steps",
                save_only_model=True,
                save_strategy="steps",

                # vLLM
                use_vllm=True,
                vllm_mode=vllm_mode,
                vllm_gpu_memory_utilization=args.train_vllm_gpu_memory_utilization,
                vllm_max_model_length=vllm_max_model_length,
                vllm_enable_sleep_mode=False,
                
                vllm_importance_sampling_correction=True,
                vllm_importance_sampling_mode= "sequence_mask" if args.importance_sampling_level == "sequence" else "token_mask",
                vllm_importance_sampling_cap=3.0,

                include_for_metrics=["reward", "entropy", "grad_norm", "epoch"]
            )
            
            # GRPO Trainer
            print(f"💰 Creating reward functions")
            
            # 복수의 reward function 리스트 생성
            reward_funcs = []
            
            # 1. 기본 reward function (embedding-based 또는 retrieval-based)
            print(f"  [1] Base reward: {args.reward_type}@{args.k}")
            if args.use_local_embedding:
                print(f"      Using local embedding-based reward calculation")
                base_reward_fn = LocalEmbeddingRewardFunction(
                    args=args,
                    uid_2_target=self.train_dataset.target_dict,
                )
            else:
                print(f"      Using RetrievalService-based reward calculation")
                base_reward_fn = RecRewardFrunction(
                    retrieval_service_name=args.retrieval_service_name,
                    namespace=args.namespace,
                    data_name=args.data_name,
                    reward_type=args.reward_type,
                    k=args.k,
                    normalize=args.normalize_rewards,
                    test_target=args.test_target,
                )
            reward_funcs.append(base_reward_fn)
            
            # 2. Similar History Item Mention Reward (옵션)
            if args.use_similar_history_reward:
                print(f"  [2] Similar History Item Mention Reward: +1.0 for mentioning similar item title (first 3 words)")
                # 임베딩 로드 (캐싱을 위해)
                emb_model_name_dir = args.emb_model_name.split("/")[-1]
                item_embedding_file_path = f"data_emb/{args.data_name}_{args.emb_type}_{emb_model_name_dir}_emb.pt"
                item_embeddings = torch.load(item_embedding_file_path, map_location=args.device)
                
                similar_history_reward_fn = SimilarHistoryItemMentionReward(
                    data_name=args.data_name,
                    item_embeddings=item_embeddings,
                    uid_2_target=self.train_dataset.target_dict,
                    device=args.device,
                    use_position_weight=args.similar_history_position_weight,
                    position_decay=args.similar_history_position_decay,
                    title_reward_weight=args.title_reward_weight,
                    brand_reward_weight=args.brand_reward_weight,
                )
                reward_funcs.append(similar_history_reward_fn)
            
            # # 3. Brand Mention Reward (옵션)
            # if args.use_brand_reward:
            #     print(f"  [3] Brand Mention Reward: +0.5 for mentioning target brand")
            #     brand_reward_fn = BrandMentionReward(
            #         data_name=args.data_name,
            #         device=args.device,
            #     )
            #     reward_funcs.append(brand_reward_fn)
            
            # # 4. Category Mention Reward (옵션)
            # if args.use_category_reward:
            #     print(f"  [4] Category Mention Reward: +0.5 for mentioning target category")
            #     category_reward_fn = CategoryMentionReward(
            #         data_name=args.data_name,
            #         device=args.device,
            #     )
            #     reward_funcs.append(category_reward_fn)
            
            # 5. Metadata Mention Reward (옵션) - 통합 메타데이터 리워드
            if args.use_metadata_reward:
                print(f"  [5] Metadata Mention Reward: +{args.metadata_base_reward} per metadata word, "
                      f"length penalty={args.metadata_length_penalty}, "
                      f"history penalty={args.history_penalty_weight}")
                metadata_reward_fn = MetadataMentionReward(
                    data_name=args.data_name,
                    device=args.device,
                    base_reward=args.metadata_base_reward,
                    length_penalty_alpha=args.metadata_length_penalty,
                    min_length=args.metadata_min_length,
                    history_penalty_weight=args.history_penalty_weight,
                )
                reward_funcs.append(metadata_reward_fn)
            
            # 6. Format Compliance Reward (옵션) - XML-like 포맷 준수
            if args.use_format_reward:
                print(f"  [6] Format Compliance Reward: +{args.format_reward_per_tag} per valid tag")
                print(f"      Required tags: {args.format_required_tags}")
                print(f"      Strict order: {args.format_strict_order}")
                format_reward_fn = FormatComplianceReward(
                    required_tags=args.format_required_tags,
                    reward_per_tag=args.format_reward_per_tag,
                    strict_order=args.format_strict_order,
                    case_sensitive=args.format_case_sensitive,
                )
                reward_funcs.append(format_reward_fn)
            
            # 7. Item Preference Mention Reward (옵션) - 메타데이터+선호도 결합
            if args.use_item_preference_reward:
                print(f"  [7] Item Preference Mention Reward: +{args.item_preference_base_reward} per common word")
                print(f"      Combines metadata + item preference text from gemma-3-1b-it")
                print(f"      Length penalty: {args.item_preference_length_penalty}")
                item_preference_reward_fn = ItemPreferenceMentionReward(
                    data_name=args.data_name,
                    device=args.device,
                    data_dir=args.data_dir,
                    data_processed_dir=args.data_processed_dir,
                    base_reward=args.item_preference_base_reward,
                    length_penalty_alpha=args.item_preference_length_penalty,
                    min_length=args.item_preference_min_length,
                )
                reward_funcs.append(item_preference_reward_fn)
            
            print(f"  Total reward functions: {len(reward_funcs)}")
            
            # reward_funcs가 1개면 단일 함수로, 2개 이상이면 리스트로 전달
            # reward_funcs_input = reward_funcs[0] if len(reward_funcs) == 1 else reward_funcs

            print(f"🎯 Initializing GRPO Trainer...")
            self.grpo_trainer = GRPOTrainerRecReward(
                model=self.model,
                args=grpo_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.valid_dataset,
                reward_funcs=reward_funcs,
                processing_class=self.tokenizer,
            )

    def evaluate_final_metrics(self, split="test"):
        """
        최종 평가: hit@k, ndcg@k (k=5,10,20)를 계산
        RecommendationEvaluator 클래스를 사용하여 평가 수행
        """
        print(f"\n{'='*80}")
        print("🧹 Cleaning up training resources before evaluation...")
        print(f"{'='*80}")

        if split == "test":
            dataset = self.test_dataset
        elif split == "valid":
            dataset = self.valid_dataset
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if hasattr(self, 'grpo_trainer'):
            del self.grpo_trainer
            self.grpo_trainer = None
            print("✓ GRPO Trainer cleaned")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 메모리 사용량 출력
        if torch.cuda.is_available():
            print("=" * 80)
            print(f"💾 GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
            print(f"💾 GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
            print("=" * 80)

        # Evaluator 인스턴스 생성 및 평가 실행
        evaluator = RecommendationEvaluator(self.args, self.args.final_checkpoint_dir)
        
        try:
            results = evaluator.evaluate(dataset, split=split, save_log=True)
        finally:
            # 평가 완료 후 메모리 정리
            evaluator.cleanup()
        
        return results
    
    def train(self):
        """
        전체 학습 루프 실행
        """
        print("=" * 80)
        print("🚀 Starting GRPO Training")
        print("=" * 80)

        if self.args.num_epochs > 0:
            self.grpo_trainer.train()
            print("=" * 80)
            print("✓ Training completed!")

            # save best model (after training, best model is loaded into grpo_trainer.model)
            os.makedirs(self.args.final_checkpoint_dir, exist_ok=True)
            self.grpo_trainer.model.save_pretrained(self.args.final_checkpoint_dir)
            print(f"✓ Best model saved: {self.args.final_checkpoint_dir}")
        else:
            print("=" * 80)
            print("⚠️  Training skipped (num_epochs=0), no model saved")
        
        # 최종 테스트 평가
        # test_metrics = self.evaluate_final_metrics(self.test_dataset, split="test")
        


def parse_args():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Recommendation System"
    )
    # basic
    parser.add_argument("--run_name", type=str, default="grpo")
    
    # Ray & Service
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--namespace", type=str, default="rl4rec")
    parser.add_argument("--retrieval_service_name", type=str, default="RetrievalService")
    parser.add_argument("--data_name", type=str, default="beauty")
    
    # Model
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_length", type=int, default=1024*4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_emb_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--use_ref_model", action="store_true", help="Use reference model for KL penalty")
    
    # Embedding Model for Evaluation
    parser.add_argument("--emb_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--emb_type", type=str, default="item_preference_1024_gemma-3-1b-it", help="Type of item text to embed (title, description, etc.)")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    
    # Prompt Generation
    parser.add_argument("--prompt_type", type=str, default="seq_rec", help="Prompt template")
    parser.add_argument("--use_brand", action="store_true", default=True, help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", help="Include category in prompt")
    parser.add_argument("--use_description", action="store_true", help="Include description in prompt")
    parser.add_argument("--use_features", action="store_true", help="Include features in prompt")
    parser.add_argument("--use_date", action="store_true", default=True, help="Include purchase date information in prompt")
    parser.add_argument("--use_relative_date", action="store_true", 
                        help="Use relative date format (D-N) based on target purchase date instead of absolute dates")
    parser.add_argument("--use_last_item", action="store_true", default=True, help="Emphasize last item")
    parser.add_argument("--emphasize_recent_item", action="store_true",
                        help="Emphasize recent purchase item with detailed information including purchase date ('This user's most recent purchase is...' format)")
    parser.add_argument("--include_target_date", action="store_true",
                        help="Include target/label item's purchase date at the end of prompt")
    parser.add_argument("--max_history_len", type=int, default=8, help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=100, help="Max words per history item")
    parser.add_argument("--days_filter", type=int, default=None,
                        help="Filter reviews to only include those within N days of target date")
    parser.add_argument("--test_target", action="store_true", help="Use target text for test")
    
    # SASRec Integration
    parser.add_argument("--use_sasrec", action="store_true",
                        help="Include SASRec recommendations in prompt as reference for query generation")
    parser.add_argument("--sasrec_top_k", type=int, default=5,
                        help="Number of top-K SASRec recommendations to include in prompt")
    
    # GRPO Training
    parser.add_argument("--loss_type", type=str, default="dr_grpo")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_sample_generations", type=int, default=8,
                        help="Number of generations per prompt for GRPO")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--train_temperature", type=float, default=0.6)
    parser.add_argument("--reference_model_kld_coef", type=float, default=0.0)
    parser.add_argument("--importance_sampling_level", type=str, default="token", choices=["token", "sequence"])

    # Reward
    parser.add_argument("--reward_type", type=str, default="ndcg", choices=["ndcg", "hit", "mixed"])
    parser.add_argument("--k", type=int, default=100, help="Top-K for metrics")
    parser.add_argument("--normalize_rewards", action="store_true", help="Normalize rewards")
    parser.add_argument("--num_negs", type=int, default=0, help="Number of negative items")
    parser.add_argument("--prepend_last_item", action="store_true",)
    
    # Multiple Reward Functions
    parser.add_argument("--use_similar_history_reward", action="store_true",
                        help="Use reward for mentioning similar history item's title (first 3 words). "
                             "Reward: +1.0 for mentioning the most similar item from purchase history.")
    parser.add_argument("--similar_history_position_weight", action="store_true",
                        help="Enable position-based weighting for similar history reward. "
                             "Earlier mentions get higher rewards.")
    parser.add_argument("--similar_history_position_decay", type=float, default=0.5,
                        help="Position decay factor for similar history reward (0.0-1.0). "
                             "0.0 = no decay (position-independent), "
                             "1.0 = full decay (reward becomes 0 at text end). "
                             "Default: 0.5 (reward halves at text end)")
    parser.add_argument("--brand_reward_weight", type=float, default=0.5,
                        help="Brand reward weight for similar history reward (default: 0.5)")
    parser.add_argument("--title_reward_weight", type=float, default=1.0,
                        help="Title reward weight for similar history reward (default: 1.0)")
    
    # Metadata Mention Reward (통합 메타데이터 리워드)
    parser.add_argument("--use_metadata_reward", action="store_true",
                        help="Use unified metadata mention reward. "
                             "Rewards mentioning metadata words (brand, category) proportionally, "
                             "with length penalty and stopword filtering using NLTK.")
    parser.add_argument("--metadata_base_reward", type=float, default=0.1,
                        help="Base reward per metadata word mentioned (default: 1.0)")
    parser.add_argument("--metadata_length_penalty", type=float, default=0.5,
                        help="Length penalty alpha for metadata reward (0.0-1.0). "
                             "Higher values penalize longer texts more (default: 0.5)")
    parser.add_argument("--metadata_min_length", type=int, default=32,
                        help="Minimum text length (in words) before length penalty applies (default: 10)")
    parser.add_argument("--history_penalty_weight", type=float, default=0.01,
                        help="Penalty weight for mentioning history metadata words not in target (default: 0.5)")
    
    # Format Compliance Reward (XML-like 포맷 준수)
    parser.add_argument("--use_format_reward", action="store_true",
                        help="Use format compliance reward. "
                             "Rewards generated text that follows the required XML-like format structure. "
                             "Example: <thinking>...</thinking> <window>...</window> <items>...</items> <query>...</query>")
    parser.add_argument("--format_required_tags", type=str, nargs="+", default=["thinking", "window", "items", "query"],
                        help="List of required XML-like tags (default: thinking window items query)")
    parser.add_argument("--format_reward_per_tag", type=float, default=0.25,
                        help="Reward per valid tag (default: 0.25). "
                             "With 4 tags, max reward = 1.0")
    parser.add_argument("--format_strict_order", action="store_true",
                        help="Enforce strict tag ordering. "
                             "If enabled and tags are out of order, reward is halved.")
    parser.add_argument("--format_case_sensitive", action="store_true",
                        help="Make tag matching case-sensitive (default: False)")
    
    # Item Preference Mention Reward (메타데이터 + 선호도 정보 결합)
    parser.add_argument("--use_item_preference_reward", action="store_true",
                        help="Use item preference mention reward. "
                             "Combines metadata (title, brand) with item preference information from gemma-3-1b-it. "
                             "Rewards mentioning common words between query and target item's combined metadata+preference. "
                             "Stopwords are automatically filtered using NLTK.")
    parser.add_argument("--item_preference_base_reward", type=float, default=0.1,
                        help="Base reward per common word mentioned (default: 0.1)")
    parser.add_argument("--item_preference_length_penalty", type=float, default=0.5,
                        help="Length penalty alpha for item preference reward (0.0-1.0). "
                             "Higher values penalize longer texts more (default: 0.5)")
    parser.add_argument("--item_preference_min_length", type=int, default=32,
                        help="Minimum text length (in words) before length penalty applies (default: 10)")
    parser.add_argument("--data_processed_dir", type=str, default="data_processed",
                        help="Directory containing processed data files like item preference JSON")


    
    # Novelty Reward (popularity-based reward)
    parser.add_argument("--novelty_reward", action="store_true",
                        help="Use novelty reward: NDCG × popularity_weight "
                             "(인기 없는 아이템을 높은 rank로 예측할수록 높은 보상)")
    parser.add_argument("--novelty_coef", type=float, default=1.0,
                        help="Novelty reward coefficient (weight for novelty reward component)")
    parser.add_argument("--novelty_target_rank", type=int, default=20,
                        help="(Deprecated, not used)")
    parser.add_argument("--novelty_mode", type=str, default="gaussian", 
                        choices=["gaussian", "uniform", "inverse"],
                        help="(Deprecated, not used)")
    parser.add_argument("--novelty_annealing", action="store_true",
                        help="Enable novelty annealing: gradually increase novelty ratio from 0 to 1 "
                             "as training progresses. Final reward = (1-ratio)*base + ratio*novelty")
    
    # Popularity Reward (long-tail item bonus)
    parser.add_argument("--popularity_coef", type=float, default=0.0,
                        help="Popularity reward coefficient (0.0 = disabled). "
                             "Rewards predicting unpopular items more.")
    
    # Target Embedding Similarity Reward
    parser.add_argument("--target_emb_reward", action="store_true",
                        help="Use target embedding similarity reward. "
                             "Rewards generated text that is similar to target item embedding.")
    parser.add_argument("--target_emb_file", type=str, default=None,
                        help="Target embedding file path. If None, uses the same embedding as emb_type.")
    parser.add_argument("--target_emb_coef", type=float, default=1.0,
                        help="Target embedding reward coefficient (weight for this reward component)")
    
    # InfoNCE Reward
    parser.add_argument("--infonce_reward", action="store_true",
                        help="Use InfoNCE (contrastive learning) reward. "
                             "Maximizes similarity with target while minimizing with negatives.")
    parser.add_argument("--infonce_coef", type=float, default=1.0,
                        help="InfoNCE reward coefficient (weight for this reward component)")
    parser.add_argument("--infonce_temperature", type=float, default=0.07,
                        help="Temperature parameter for InfoNCE (default: 0.07)")
    parser.add_argument("--infonce_emb_type", type=str, default=None,
                        help="Embedding type for InfoNCE (e.g., 'title_emb'). "
                             "If None, uses the same embedding as emb_type.")
    
    # Proxy Label Reward
    parser.add_argument("--proxy_label_reward", action="store_true",
                        help="Use proxy label reward: treats items similar to target as soft labels. "
                             "Rewards predicting target-similar items proportional to their similarity.")
    parser.add_argument("--proxy_k", type=int, default=100,
                        help="Number of proxy items (similar items) to use as soft labels")
    parser.add_argument("--proxy_label_coef", type=float, default=1.0,
                        help="Proxy label reward coefficient (weight for this reward component)")
    parser.add_argument("--proxy_label_cutoff", type=float, default=0.1,)
    parser.add_argument("--proxy_label_file", type=str, default=None,
                        help="Path to pre-computed proxy labels JSON file. "
                             "If None, automatically constructs path from data_name, emb_type, and emb_model_name. "
                             "Example: data_emb/beauty_proxy_labels_k1000_random_th0.3_item_preference_1024_gemma-3-4b-it_mxbai-embed-large-v1.json")
    
    # Anchor-Guided GRPO (AG-GRPO)
    parser.add_argument("--anchor_reward", action="store_true",
                        help="Use Anchor-Guided GRPO reward: guides exploration by anchoring to last item. "
                             "Rewards queries similar to user's last item, with dynamic radius control.")
    parser.add_argument("--anchor_coef", type=float, default=1.0,
                        help="Anchor reward coefficient (weight for this reward component)")
    parser.add_argument("--anchor_radius_start", type=float, default=0.5,
                        help="Initial exploration radius (cosine similarity threshold) at training start. "
                             "Lower values = narrower exploration around last item.")
    parser.add_argument("--anchor_radius_end", type=float, default=1.0,
                        help="Final exploration radius at training end. Gradually increases via curriculum learning.")
    parser.add_argument("--anchor_penalty_mode", type=str, default="soft", choices=["soft", "hard"],
                        help="Penalty mode for out-of-radius queries. "
                             "'soft': uses similarity as reward regardless of radius. "
                             "'hard': applies penalty if similarity < current_radius.")
    parser.add_argument("--anchor_penalty_value", type=float, default=-1.0,
                        help="Penalty value for hard mode when query is outside the radius")
    
    # Adaptive Threshold Reward
    parser.add_argument("--adaptive_threshold_reward", action="store_true",
                        help="Use Adaptive Threshold Reward: dynamic threshold based on historical item similarity. "
                             "Rewards = 1 if CosSim(query, target) > max(tau_min, S_base), else 0. "
                             "S_base = mean similarity between query and historical items. "
                             "Enforces strict criteria: 'query must be more similar to target than to past purchases'.")
    parser.add_argument("--adaptive_threshold_coef", type=float, default=1.0,
                        help="Adaptive threshold reward coefficient (weight for this reward component)")
    parser.add_argument("--adaptive_tau_min", type=float, default=0.0,
                        help="Minimum threshold (tau_min) for adaptive threshold. "
                             "Acts as a floor value when S_base is very low. "
                             "threshold = max(tau_min, S_base)")
    
    # History Proxy Threshold Reward
    parser.add_argument("--history_proxy_threshold_reward", action="store_true",
                        help="Use History Proxy Threshold Reward: uses most similar history item to target as proxy. "
                             "Pre-computes proxy = argmax_i CosSim(target, history_i) for each user. "
                             "Reward = max(0, CosSim(query, proxy) - mean(CosSim(query, other_history))). "
                             "Rewards query similarity to best matching history item above average baseline.")
    parser.add_argument("--history_proxy_threshold_coef", type=float, default=0.2,
                        help="History proxy threshold reward coefficient (weight for this reward component)")
    
    # Local Embedding-based Reward (alternative to RetrievalService)
    parser.add_argument("--use_local_embedding", action="store_true", 
                        help="Use local embedding-based reward instead of RetrievalService")
    parser.add_argument("--emb_batch_size", type=int, default=128,
                        help="Batch size for embedding computation")
    
    # Logging & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--final_checkpoint_dir", type=str, default="checkpoints/grpo/checkpoint-5000")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--report_to", type=str, default="wandb", help="Logging backend (wandb)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    # vllm for evaluation
    parser.add_argument("--train_vllm_gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_max_tokens", type=int, default=512)
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100000)
    parser.add_argument("--dummy_generation", action="store_true", help="Use dummy generation")
    
    # vLLM Server Mode (for separate GPU execution)
    parser.add_argument("--use_vllm_server", action="store_true",
                        help="Use external vLLM server instead of colocated vLLM. "
                             "This allows running vLLM on a separate GPU.")
    parser.add_argument("--vllm_server_url", type=str, default="http://localhost:8000",
                        help="URL of the vLLM server (default: http://localhost:8000)")
    parser.add_argument("--vllm_gpu_id", type=int, default=0,
                        help="GPU ID for vLLM server (used in startup script)")
    parser.add_argument("--train_gpu_id", type=int, default=1,
                        help="GPU ID for GRPO training (used when use_vllm_server is enabled)")
    
    # Rank-based filtering for training
    parser.add_argument("--filter_train_csv", type=str, default=None,
                        help="Path to evaluation CSV file for filtering train set by rank")
    parser.add_argument("--rank_min", type=int, default=None,
                        help="Minimum rank for filtering (inclusive, None = no limit)")
    parser.add_argument("--rank_max", type=int, default=None,
                        help="Maximum rank for filtering (inclusive, None = no limit)")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    args.sequential_file = f"data/{args.data_name}/sequential_data.txt"

    args.eval_interval = args.save_interval

    # fix seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # vLLM 로깅 레벨 조정 (INFO 메시지 억제)
    std_logging.getLogger("vllm").setLevel(std_logging.ERROR)
    
    # Trainer 초기화 및 학습
    trainer = GRPOTrainerWrapper(args)
    trainer.train()


if __name__ == "__main__":
    main()
