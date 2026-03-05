#!/usr/bin/env python3
"""
GRPO Training Script for Rating Prediction
Rating 예측 정확도를 리워드로 사용하는 GRPO 학습
리워드: R = 1 - |y_hat - y| / 4  (정확하면 1, 갭이 4이면 0)
"""

import os
import re
import gc
import json
import torch
import random
import argparse
import numpy as np
import logging as std_logging
from typing import List, Dict, Optional, Tuple

from trl import GRPOTrainer, GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from accelerate import logging
from accelerate.utils import gather
import wandb

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RATING_PROMPT_HEAD = (
    "You are a helpful assistant that predicts product ratings.\n"
    "A user has the following purchase history (ordered from oldest to most recent).\n"
    "Predict what rating (1–5 stars) the user would give to the target item.\n"
)

RATING_PROMPT_TAIL = (
    "Briefly reason about the user's preferences based on their history, "
    "then output exactly: rating: X  (where X is an integer from 1 to 5)."
)


def build_rating_prompt(
    history_reviews: List[Dict],
    target_item_meta: Dict,
    item_metadata: Dict,
    max_history_len: int = 8,
    tokenizer=None,
) -> str:
    """
    히스토리 리뷰와 타겟 아이템 메타데이터로 프롬프트 생성.
    리뷰 텍스트는 포함하지 않고 아이템 메타데이터 + 날짜 + 별점만 사용.

    Args:
        history_reviews: 타겟 이전의 리뷰 리스트 (각 원소는 user2reviews_with_date.json 엔트리)
        target_item_meta: 타겟 아이템의 메타데이터 딕셔너리
        item_metadata: 전체 아이템 메타데이터 {item_id: {...}}
        max_history_len: 최대 히스토리 길이
        tokenizer: chat template 적용용 (선택)
    """
    # 최근 max_history_len개만 사용
    history_reviews = history_reviews[-max_history_len:]

    history_lines = []
    for i, review in enumerate(history_reviews):
        item_id = int(review.get("item_id", -1))
        meta = item_metadata.get(item_id, {})
        title = meta.get("title", "Unknown Item")
        brand = meta.get("brand", "")
        date = review.get("date", "")
        rating = review.get("rating", "")

        line = f"{i+1}. {title}"
        if brand:
            line += f" (Brand: {brand})"
        if date:
            line += f" [{date}]"
        if rating:
            line += f" — {rating} stars"
        history_lines.append(line)

    history_text = "\n".join(history_lines) if history_lines else "(No purchase history)"

    target_title = target_item_meta.get("title", "Unknown Item")
    target_brand = target_item_meta.get("brand", "")
    target_category = target_item_meta.get("category", "")

    target_lines = [f"Title: {target_title}"]
    if target_brand:
        target_lines.append(f"Brand: {target_brand}")
    if target_category:
        target_lines.append(f"Category: {target_category}")
    target_text = "\n".join(target_lines)

    prompt = (
        f"{RATING_PROMPT_HEAD}\n"
        f"--- Purchase History ---\n"
        f"{history_text}\n\n"
        f"--- Target Item ---\n"
        f"{target_text}\n\n"
        f"{RATING_PROMPT_TAIL}\n"
    )

    if tokenizer is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RatingPredictionDataset(Dataset):
    """
    Rating 예측용 데이터셋.

    sequential_data.txt 에서 train/valid/test 분할을 따르고,
    user2reviews_with_date.json 에서 타겟 아이템의 실제 rating을 가져온다.
    """

    def __init__(
        self,
        data_name: str,
        item_metadata: Dict,
        tokenizer=None,
        split: str = "train",
        max_history_len: int = 8,
        data_dir: str = "data",
    ):
        self.data_name = data_name
        self.item_metadata = item_metadata
        self.tokenizer = tokenizer
        self.split = split
        self.max_history_len = max_history_len

        # sequential_data.txt 로드
        sequential_file = os.path.join(data_dir, data_name, "sequential_data.txt")
        if split == "train":
            target_index = -3
        elif split == "valid":
            target_index = -2
        elif split == "test":
            target_index = -1
        else:
            raise ValueError(f"Invalid split: {split}")

        self.user_ids: List[int] = []
        self.history_dict: Dict[int, List[int]] = {}
        self.target_dict: Dict[int, int] = {}

        with open(sequential_file, "r") as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:target_index]
                target = parts[target_index]
                self.user_ids.append(user_id)
                self.history_dict[user_id] = history
                self.target_dict[user_id] = target

        # user2reviews_with_date.json 로드
        reviews_file = os.path.join(data_dir, data_name, "user2reviews_with_date.json")
        with open(reviews_file, "r") as f:
            self.user_reviews: Dict[str, List[Dict]] = json.load(f)
        print(f"  Loaded reviews for {len(self.user_reviews)} users from {reviews_file}")

        # 아이템 ID → 리뷰 매핑 (per user) 및 프롬프트/레이팅 캐싱
        print(f"  Building prompts and ratings for {split} split ({len(self.user_ids)} users)...")
        self.prompts: Dict[int, str] = {}
        self.ratings: Dict[int, float] = {}  # 타겟 아이템의 실제 rating

        skipped = 0
        for user_id in self.user_ids:
            target_item_id = self.target_dict[user_id]
            reviews = self.user_reviews.get(str(user_id), [])

            # item_id → review 매핑
            item_to_review: Dict[int, Dict] = {
                int(r.get("item_id", -1)): r for r in reviews
            }

            # 타겟 아이템 rating 가져오기
            target_review = item_to_review.get(target_item_id)
            if target_review is None or target_review.get("rating") is None:
                skipped += 1
                self.ratings[user_id] = None
            else:
                self.ratings[user_id] = float(target_review["rating"])

            # 히스토리 아이템들의 리뷰 (타겟 이전 것들)
            history_item_ids = self.history_dict[user_id]
            history_reviews = [
                item_to_review[iid]
                for iid in history_item_ids
                if iid in item_to_review
            ]

            # 타겟 아이템 메타데이터
            target_meta = self.item_metadata.get(target_item_id, {})

            self.prompts[user_id] = build_rating_prompt(
                history_reviews=history_reviews,
                target_item_meta=target_meta,
                item_metadata=self.item_metadata,
                max_history_len=self.max_history_len,
                tokenizer=self.tokenizer,
            )

        # rating이 없는 샘플 제외
        self.user_ids = [
            uid for uid in self.user_ids if self.ratings.get(uid) is not None
        ]
        print(f"  {split.upper()} dataset: {len(self.user_ids)} samples "
              f"(skipped {skipped} without rating)")

        # 샘플 프롬프트 출력
        if self.user_ids:
            sample_uid = self.user_ids[0]
            print(f"\n--- Sample prompt (user {sample_uid}) ---")
            print(self.prompts[sample_uid][:800])
            print(f"  -> target rating: {self.ratings[sample_uid]}")
            print("-" * 60)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return {
            "prompt": self.prompts[user_id],
            "target_rating": self.ratings[user_id],
            "user_id": user_id,
            # GRPOTrainer requires 'history' and 'target' keys in reward kwargs
            "history": self.history_dict[user_id],
            "target": self.target_dict[user_id],
        }


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

class RatingAccuracyReward:
    """
    Rating 예측 정확도 기반 리워드.
    R = 1 - |y_hat - y| / 4
    정확하면 1, 갭이 4이면 0 (선형 보간).
    """

    # GRPOTrainer가 reward function의 __name__을 요구함
    __name__ = "RatingAccuracyReward"

    def __init__(self):
        self._last_parsed_ratings: List[Optional[float]] = []

    def _parse_rating(self, text: str) -> Optional[float]:
        """'rating: X' 패턴에서 숫자 추출 (마지막 매칭 우선)."""
        # "rating: 4" or "rating:4" or "Rating: 3.5"
        matches = re.findall(r'rating\s*:\s*([1-5](?:\.\d+)?)', text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        # fallback: 단독 숫자 1-5 찾기
        matches = re.findall(r'\b([1-5])\b', text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return None

    def __call__(
        self,
        generated_texts: List[str],
        targets: List,           # target item id (unused for rating)
        histories: List,         # history item ids (unused for rating)
        user_ids: List,          # user ids (unused)
        target_rating: List[float],  # 실제 rating 값
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = []
        self._last_parsed_ratings = []

        for text, true_rating in zip(generated_texts, target_rating):
            predicted = self._parse_rating(text)
            self._last_parsed_ratings.append(predicted)

            if predicted is None:
                # 파싱 실패 → 최대 패널티
                rewards.append(0.0)
            else:
                # R = 1 - |y_hat - y| / 4  (클리핑하여 [0, 1] 범위)
                gap = abs(predicted - true_rating)
                reward = max(0.0, 1.0 - gap / 4.0)
                rewards.append(reward)

        return rewards


# ---------------------------------------------------------------------------
# GRPOTrainer 서브클래스
# ---------------------------------------------------------------------------

class GRPOTrainerRatingReward(GRPOTrainer):
    """
    Rating 예측 리워드를 위한 GRPOTrainer 서브클래스.
    target_rating 을 reward_kwargs로 전달할 수 있도록 _calculate_rewards 오버라이드.
    """

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
                strict=True,
            )
        ):
            with profiling_context(self, reward_func_name):
                output_reward_func = reward_func(
                    generated_texts=completions,
                    targets=reward_kwargs["target"],
                    histories=reward_kwargs["history"],
                    user_ids=reward_kwargs["user_id"],
                    target_rating=reward_kwargs["target_rating"],
                    **{
                        k: v
                        for k, v in reward_kwargs.items()
                        if k not in ["target", "history", "user_id", "target_rating", "trainer_state"]
                    },
                )
                output_reward_func = [
                    r if r is not None else torch.nan for r in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_kwargs = {
                key: value[nan_row_idx]
                for key, value in reward_kwargs.items()
                if key != "trainer_state"
            }
            row_kwargs["prompt"] = prompts[nan_row_idx]
            row_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func


# ---------------------------------------------------------------------------
# Trainer Wrapper
# ---------------------------------------------------------------------------

class RatingGRPOTrainerWrapper:
    """Rating 예측 GRPO 학습기."""

    def __init__(self, args):
        self.args = args

        # 토크나이저 로드
        print(f"Loading tokenizer: {args.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        print(f"Loading model: {args.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True
        )

        # 아이템 메타데이터 로드
        print(f"Loading item metadata...")
        self.item_metadata = self._load_item_metadata(args.data_name, args.data_dir)

        # 데이터셋 생성
        print(f"Creating datasets...")
        dataset_kwargs = dict(
            data_name=args.data_name,
            item_metadata=self.item_metadata,
            tokenizer=self.tokenizer,
            max_history_len=args.max_history_len,
            data_dir=args.data_dir,
        )
        self.train_dataset = RatingPredictionDataset(split="train", **dataset_kwargs)
        self.valid_dataset = RatingPredictionDataset(split="valid", **dataset_kwargs)

        if args.num_epochs > 0:
            if args.report_to == "wandb":
                print(f"Initializing Weights & Biases...")
                wandb.init(
                    project="rl4rec",
                    name=args.run_name,
                    config=vars(args),
                )
                print(f"Wandb initialized")

            vllm_mode = "server" if args.use_vllm_server else "colocate"
            vllm_max_model_length = args.max_length + args.max_new_tokens

            if args.use_vllm_server:
                args.train_vllm_gpu_memory_utilization = None
                vllm_max_model_length = None

            grpo_config = GRPOConfig(
                output_dir=args.checkpoint_dir,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                logging_steps=args.log_interval,
                save_steps=args.save_interval,
                save_total_limit=args.save_total_limit,
                dataloader_num_workers=4,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                bf16=True,
                report_to=args.report_to if args.report_to != "none" else None,
                run_name=args.run_name,
                beta=args.reference_model_kld_coef,
                loss_type=args.loss_type,
                # GRPO specific
                num_generations=args.num_sample_generations,
                temperature=args.train_temperature,
                max_steps=args.max_steps,
                # Generation
                max_completion_length=args.max_new_tokens,
                repetition_penalty=1.0,
                top_p=0.95,
                # Save
                save_only_model=True,
                save_strategy="steps",
                # vLLM
                use_vllm=True,
                vllm_mode=vllm_mode,
                vllm_gpu_memory_utilization=args.train_vllm_gpu_memory_utilization,
                vllm_max_model_length=vllm_max_model_length,
                vllm_enable_sleep_mode=False,
                vllm_importance_sampling_correction=True,
                vllm_importance_sampling_mode=(
                    "sequence_mask"
                    if args.importance_sampling_level == "sequence"
                    else "token_mask"
                ),
                vllm_importance_sampling_cap=3.0,
                include_for_metrics=["reward", "entropy", "grad_norm", "epoch"],
            )

            print(f"Creating reward function: RatingAccuracyReward")
            reward_fn = RatingAccuracyReward()

            print(f"Initializing GRPO Trainer...")
            self.grpo_trainer = GRPOTrainerRatingReward(
                model=self.model,
                args=grpo_config,
                train_dataset=self.train_dataset,
                eval_dataset=self.valid_dataset,
                reward_funcs=[reward_fn],
                processing_class=self.tokenizer,
            )

    def _load_item_metadata(self, data_name: str, data_dir: str = "data") -> Dict:
        path = os.path.join(data_dir, data_name, "meta_text_fix.json")
        if os.path.exists(path):
            print(f"  Loading item metadata from: {path}")
            with open(path, "r") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        print(f"  WARNING: Item metadata not found at {path}. Using empty metadata.")
        return {}

    def train(self):
        print("=" * 80)
        print("Starting GRPO Rating Prediction Training")
        print("=" * 80)

        if self.args.num_epochs > 0:
            self.grpo_trainer.train()
            print("Training completed!")

            os.makedirs(self.args.final_checkpoint_dir, exist_ok=True)
            self.grpo_trainer.model.save_pretrained(self.args.final_checkpoint_dir)
            print(f"Model saved: {self.args.final_checkpoint_dir}")
        else:
            print("Training skipped (num_epochs=0)")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO Training for Rating Prediction"
    )
    parser.add_argument("--run_name", type=str, default="grpo_rating")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--data_dir", type=str, default="data")

    # Model
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Prompt
    parser.add_argument("--max_history_len", type=int, default=8)

    # GRPO Training
    parser.add_argument("--loss_type", type=str, default="dr_grpo")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument(
        "--num_sample_generations",
        type=int,
        default=8,
        help="Number of generations per prompt for GRPO",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--train_temperature", type=float, default=0.8)
    parser.add_argument("--reference_model_kld_coef", type=float, default=0.0)
    parser.add_argument(
        "--importance_sampling_level",
        type=str,
        default="token",
        choices=["token", "sequence"],
    )

    # Logging & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grpo_rating")
    parser.add_argument(
        "--final_checkpoint_dir",
        type=str,
        default="checkpoints/grpo_rating/final",
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--report_to", type=str, default="wandb", help="Logging backend (wandb/none)"
    )

    # vLLM
    parser.add_argument(
        "--train_vllm_gpu_memory_utilization", type=float, default=0.45
    )
    parser.add_argument("--use_vllm_server", action="store_true")
    parser.add_argument(
        "--vllm_server_url", type=str, default="http://localhost:8000"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    std_logging.getLogger("vllm").setLevel(std_logging.ERROR)

    trainer = RatingGRPOTrainerWrapper(args)
    trainer.train()


if __name__ == "__main__":
    main()
