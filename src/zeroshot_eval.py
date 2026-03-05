#!/usr/bin/env python3
"""
Zeroshot Evaluation Script with Multiple Rollouts
Evaluates model without training, generating multiple rollouts per prompt
"""

import torch
import argparse
import gc
import os
import json
import pandas as pd
from tqdm import tqdm
import random
from typing import List, Dict, Tuple
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.common import (
    load_json,
    print_arguments,
    initialize_tokenizer,
)
from utils.dataset import create_dataloaders, load_item_metadata
from evaluator import RecommendationEvaluator

import warnings
warnings.filterwarnings("ignore")


def load_trigger_items(trigger_items_dir: str, data_name: str, split: str = "test") -> Dict[int, int]:
    """
    Load trigger items from JSON file
    
    Args:
        trigger_items_dir: Directory containing trigger items files
        data_name: Dataset name (e.g., 'beauty', 'toys')
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        trigger_items_dict: {user_id: trigger_item_id} dictionary
    """
    # Convert 'valid' to 'val' for file naming
    file_split = 'val' if split == 'valid' else split
    
    trigger_file = f"{trigger_items_dir}/SASRec_{data_name}_{file_split}_trigger_items.json"
    print(f"📌 Loading trigger items from: {trigger_file}")
    
    if not os.path.exists(trigger_file):
        print(f"⚠️  Warning: Trigger items file not found: {trigger_file}")
        return {}
    
    with open(trigger_file, 'r') as f:
        trigger_items = json.load(f)
    
    # Convert keys to int
    trigger_items_dict = {int(k): int(v) for k, v in trigger_items.items()}
    print(f"✓ Loaded trigger items for {len(trigger_items_dict)} users")
    
    return trigger_items_dict


def generate_rollouts_with_vllm(
    llm: LLM,
    prompts: List[str],
    user_ids: List[int],
    num_rollouts: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 128,
    top_p: float = 0.9,
    top_k: int = -1,
) -> List[Dict]:
    """
    Generate multiple rollouts for each prompt using vLLM
    
    Args:
        llm: vLLM model instance
        prompts: List of prompt strings
        user_ids: List of user IDs
        num_rollouts: Number of rollouts per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
    
    Returns:
        List of dictionaries with user_id, rollout_index, and generated_text
    """
    print(f"\n{'='*80}")
    print(f"🎲 Generating {num_rollouts} rollout(s) per prompt")
    print(f"{'='*80}")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Top-p: {top_p}")
    print(f"  Top-k: {top_k}")
    print(f"{'='*80}\n")
    
    # Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n=num_rollouts,  # Generate num_rollouts outputs per prompt
    )
    
    # Generate outputs
    print("🔄 Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect results
    results = []
    for i, output in enumerate(tqdm(outputs, desc="Processing outputs")):
        user_id = user_ids[i]
        
        # Each output has multiple completions (rollouts)
        for rollout_idx, completion in enumerate(output.outputs):
            result = {
                'user_id': user_id,
                'rollout_index': rollout_idx,
                'generated_text': completion.text.strip(),
                'prompt': prompts[i],
            }
            results.append(result)
    
    print(f"✓ Generated {len(results)} total outputs ({len(prompts)} prompts × {num_rollouts} rollouts)")    
    return results


def calculate_rollout_similarities(
    generation_results: List[Dict],
    evaluator: RecommendationEvaluator,
) -> List[Dict]:
    """
    Calculate cosine similarity between rollouts from the same prompt/user
    
    Args:
        generation_results: List of generation results with user_id and generated_text
        evaluator: RecommendationEvaluator instance with embedding model
    
    Returns:
        Updated generation_results with avg_rollout_similarity field
    """
    print(f"\n{'='*80}")
    print(f"📏 Calculating Rollout Similarities")
    print(f"{'='*80}")
    
    # Group results by user_id
    user_rollouts = {}
    for result in generation_results:
        user_id = result['user_id']
        if user_id not in user_rollouts:
            user_rollouts[user_id] = []
        user_rollouts[user_id].append(result)
    
    # Extract all texts for embedding
    all_texts = [r['generated_text'] for r in generation_results]
    
    print(f"  Computing embeddings for {len(all_texts)} texts...")
    
    # Get embeddings using evaluator's embedding model
    embeddings = evaluator.compute_embeddings(all_texts)
    
    # Convert to numpy if it's a torch tensor
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()
    
    # Create a mapping from (user_id, rollout_index) to embedding
    embedding_map = {}
    for i, result in enumerate(generation_results):
        key = (result['user_id'], result['rollout_index'])
        embedding_map[key] = embeddings[i]
    
    print(f"  Calculating similarities for {len(user_rollouts)} users...")
    
    # Calculate average similarity for each rollout
    for user_id, user_results in tqdm(user_rollouts.items(), desc="Computing similarities"):
        if len(user_results) <= 1:
            # Only one rollout, no similarity to compute
            for result in user_results:
                result['avg_rollout_similarity'] = 1.0
            continue
        
        # Get embeddings for this user's rollouts
        user_embeddings = []
        for result in user_results:
            key = (result['user_id'], result['rollout_index'])
            user_embeddings.append(embedding_map[key])
        
        user_embeddings = np.array(user_embeddings)
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(user_embeddings)
        
        # For each rollout, compute average similarity with other rollouts
        for i, result in enumerate(user_results):
            # Exclude self-similarity (diagonal element)
            other_sims = np.concatenate([similarities[i, :i], similarities[i, i+1:]])
            avg_sim = float(np.mean(other_sims)) if len(other_sims) > 0 else 1.0
            result['avg_rollout_similarity'] = avg_sim
    
    print(f"✓ Calculated rollout similarities")
    
    # Print summary statistics
    all_sims = [r['avg_rollout_similarity'] for r in generation_results if 'avg_rollout_similarity' in r]
    if all_sims:
        print(f"  Average similarity: {np.mean(all_sims):.4f} ± {np.std(all_sims):.4f}")
        print(f"  Min: {np.min(all_sims):.4f}, Max: {np.max(all_sims):.4f}")
    
    return generation_results


def evaluate_rollouts(
    args,
    results: List[Dict],
    dataset,
    item_metadata: Dict,
    evaluator: RecommendationEvaluator,
    split: str = "test",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Evaluate each rollout and return results as DataFrame
    
    Args:
        args: Arguments
        results: List of generation results
        dataset: Dataset to evaluate
        item_metadata: Item metadata dictionary
        evaluator: RecommendationEvaluator instance
        split: Dataset split name
    
    Returns:
        Tuple of (DataFrame with evaluation metrics for each rollout, 
                  List of rank information for each sample in each rollout)
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating Rollouts")
    print(f"{'='*80}")
    
    # Group results by user_id
    user_rollouts = {}
    for result in results:
        user_id = result['user_id']
        if user_id not in user_rollouts:
            user_rollouts[user_id] = []
        user_rollouts[user_id].append(result)
    
    # Evaluate each rollout
    eval_results = []
    all_rank_info = []  # Store rank info for all rollouts
    
    # Get unique rollout indices
    num_rollouts = max([r['rollout_index'] for r in results]) + 1
    
    for rollout_idx in range(num_rollouts):
        print(f"\n--- Evaluating Rollout {rollout_idx} ---")
        
        # Collect generated texts for this rollout
        generated_texts = []
        rollout_user_ids = []
        
        for user_id in sorted(user_rollouts.keys()):
            user_results = user_rollouts[user_id]
            # Find the result for this rollout index
            rollout_result = next((r for r in user_results if r['rollout_index'] == rollout_idx), None)
            if rollout_result:
                generated_texts.append(rollout_result['generated_text'])
                rollout_user_ids.append(user_id)
        
        print(f"  Evaluating {len(generated_texts)} samples for rollout {rollout_idx}...")
        
        # Run evaluation for this rollout
        try:
            # Get metrics and rank_info from evaluator
            # Note: evaluator.evaluate() needs to be modified to return rank_info
            # For now, we'll call compute_retrieval_metrics directly
            
            # First, get the data we need
            all_prompts = []
            all_targets = []
            all_histories = []
            
            for i in range(len(dataset)):
                sample = dataset[i]
                all_prompts.append(sample["prompt"])
                all_targets.append(sample["target"])
                all_histories.append(sample["history"])
                if i >= len(generated_texts) - 1:
                    break
            
            # Compute embeddings for generated texts
            query_embeddings = evaluator.compute_embeddings(generated_texts)
            
            # Compute retrieval metrics (returns metrics and rank_info)
            metrics, rank_info = evaluator.compute_retrieval_metrics(
                query_embeddings, 
                all_targets[:len(generated_texts)], 
                all_histories[:len(generated_texts)],
                ks=[5, 10, 20]
            )
            
            # Add rollout index and user_id to rank_info
            for i, rank_data in enumerate(rank_info):
                rank_data['rollout_index'] = rollout_idx
                rank_data['user_id'] = rollout_user_ids[i]
            
            all_rank_info.extend(rank_info)
            
            # Average metrics
            averaged_metrics = {}
            for key, values in metrics.items():
                if isinstance(values, list):
                    averaged_metrics[key] = sum(values) / len(values) if len(values) > 0 else 0.0
                else:
                    averaged_metrics[key] = values
            
            # Add rollout index to metrics
            averaged_metrics['rollout_index'] = rollout_idx
            eval_results.append(averaged_metrics)
            
            print(f"  Rollout {rollout_idx} results:")
            for metric_name, value in averaged_metrics.items():
                if metric_name != 'rollout_index':
                    print(f"    {metric_name}: {value:.4f}")
        
        except Exception as e:
            print(f"  ❌ Error evaluating rollout {rollout_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(eval_results)
    
    print(f"\n{'='*80}")
    print("✅ Evaluation Complete!")
    print(f"{'='*80}")
    
    return results_df, all_rank_info


def save_results(
    generation_results: List[Dict],
    eval_results_df: pd.DataFrame,
    rank_info: List[Dict],
    args,
    split: str = "test",
    output_dir: str = "results",
):
    """
    Save generation and evaluation results to CSV files
    
    Args:
        generation_results: List of generation results
        eval_results_df: DataFrame with evaluation metrics
        rank_info: List of rank information for each sample in each rollout
        args: Arguments
        split: Dataset split name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename without date
    base_name = f"zeroshot_{args.data_name}_{args.prompt_type}_{args.num_rollouts}_{split}"
    if args.use_trigger_items:
        base_name += "_trigger"
    
    # Merge generation results with rank information
    gen_df = pd.DataFrame(generation_results)
    rank_df = pd.DataFrame(rank_info)
    
    # Merge on user_id and rollout_index to add rank columns to generation results
    if len(rank_df) > 0:
        gen_df = gen_df.merge(
            rank_df[['user_id', 'rollout_index', 'target_item', 'target_score', 'rank', 'is_cold', 'target_novelty']], 
            on=['user_id', 'rollout_index'],
            how='left'
        )
    
    # Save generation results with rank information
    gen_csv_path = f"{output_dir}/{base_name}_generations.csv"
    gen_df.to_csv(gen_csv_path, index=False)
    print(f"\n💾 Saved generation results (with rank info) to: {gen_csv_path}")
    
    # Save evaluation results
    eval_csv_path = f"{output_dir}/{base_name}_eval.csv"
    eval_results_df.to_csv(eval_csv_path, index=False)
    print(f"💾 Saved evaluation results to: {eval_csv_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("📈 Summary Statistics Across Rollouts")
    print(f"{'='*80}")
    
    # Calculate mean and std for each metric
    metric_cols = [col for col in eval_results_df.columns if col != 'rollout_index']
    for col in metric_cols:
        mean_val = eval_results_df[col].mean()
        std_val = eval_results_df[col].std()
        print(f"{col:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print rank statistics
    if len(rank_df) > 0:
        print(f"\n{'='*80}")
        print("📊 Rank Statistics")
        print(f"{'='*80}")
        print(f"Average rank: {rank_df['rank'].mean():.2f}")
        print(f"Median rank: {rank_df['rank'].median():.2f}")
        print(f"Min rank: {rank_df['rank'].min()}")
        print(f"Max rank: {rank_df['rank'].max()}")
        
        # Per-rollout rank statistics
        print(f"\n{'='*80}")
        print("📊 Rank Statistics by Rollout")
        print(f"{'='*80}")
        for rollout_idx in sorted(rank_df['rollout_index'].unique()):
            rollout_ranks = rank_df[rank_df['rollout_index'] == rollout_idx]['rank']
            print(f"Rollout {rollout_idx}: mean={rollout_ranks.mean():.2f}, "
                  f"median={rollout_ranks.median():.2f}, "
                  f"min={rollout_ranks.min()}, max={rollout_ranks.max()}")
        
        # Per-user rank statistics: difference between mean and min rank
        print(f"\n{'='*80}")
        print("📊 Per-User Rank Analysis: Mean - Min Difference")
        print(f"{'='*80}")
        
        user_rank_diffs = []
        for user_id in rank_df['user_id'].unique():
            user_ranks = rank_df[rank_df['user_id'] == user_id]['rank']
            mean_rank = user_ranks.mean()
            min_rank = user_ranks.min()
            max_rank = user_ranks.max()
            rank_diff = mean_rank - min_rank
            user_rank_diffs.append({
                'user_id': user_id,
                'mean_rank': mean_rank,
                'min_rank': min_rank,
                'max_rank': max_rank,
                'diff': rank_diff
            })
        
        # Calculate average difference for all users
        avg_rank_diff = np.mean([d['diff'] for d in user_rank_diffs])
        print(f"Average (Mean Rank - Min Rank) across all users: {avg_rank_diff:.2f}")
        print(f"Number of users analyzed: {len(user_rank_diffs)}")
        
        # Additional statistics for all users
        rank_diffs = [d['diff'] for d in user_rank_diffs]
        print(f"Std of rank differences: {np.std(rank_diffs):.2f}")
        print(f"Min difference: {np.min(rank_diffs):.2f}")
        print(f"Max difference: {np.max(rank_diffs):.2f}")
        print(f"Median difference: {np.median(rank_diffs):.2f}")
        
        # Filter users with min_rank <= 100
        filtered_user_diffs_le100 = [d for d in user_rank_diffs if d['min_rank'] <= 100]
        
        if len(filtered_user_diffs_le100) > 0:
            print(f"\n{'='*80}")
            print("📊 Per-User Rank Analysis (Min Rank ≤ 100 only)")
            print(f"{'='*80}")
            
            filtered_rank_diffs = [d['diff'] for d in filtered_user_diffs_le100]
            filtered_avg_rank_diff = np.mean(filtered_rank_diffs)
            
            print(f"Average (Mean Rank - Min Rank) for users with min_rank ≤ 100: {filtered_avg_rank_diff:.2f}")
            print(f"Number of users with min_rank ≤ 100: {len(filtered_user_diffs_le100)}")
            print(f"Percentage of total users: {len(filtered_user_diffs_le100) / len(user_rank_diffs) * 100:.1f}%")
            print(f"Std of rank differences: {np.std(filtered_rank_diffs):.2f}")
            print(f"Min difference: {np.min(filtered_rank_diffs):.2f}")
            print(f"Max difference: {np.max(filtered_rank_diffs):.2f}")
            print(f"Median difference: {np.median(filtered_rank_diffs):.2f}")
            
            # Save filtered results to separate CSV
            filtered_user_rank_diff_df = pd.DataFrame(filtered_user_diffs_le100)
            filtered_diff_csv_path = f"{output_dir}/{base_name}_user_rank_diff_min100.csv"
            filtered_user_rank_diff_df.to_csv(filtered_diff_csv_path, index=False)
            print(f"\n💾 Saved filtered per-user rank differences (min_rank ≤ 100) to: {filtered_diff_csv_path}")
        else:
            print(f"\n⚠️  No users found with min_rank ≤ 100")
        
        # Filter users with min_rank > 100
        filtered_user_diffs_gt100 = [d for d in user_rank_diffs if d['min_rank'] > 100]
        
        if len(filtered_user_diffs_gt100) > 0:
            print(f"\n{'='*80}")
            print("📊 Per-User Rank Analysis (Min Rank > 100 only)")
            print(f"{'='*80}")
            
            filtered_rank_diffs_gt = [d['diff'] for d in filtered_user_diffs_gt100]
            filtered_avg_rank_diff_gt = np.mean(filtered_rank_diffs_gt)
            
            print(f"Average (Mean Rank - Min Rank) for users with min_rank > 100: {filtered_avg_rank_diff_gt:.2f}")
            print(f"Number of users with min_rank > 100: {len(filtered_user_diffs_gt100)}")
            print(f"Percentage of total users: {len(filtered_user_diffs_gt100) / len(user_rank_diffs) * 100:.1f}%")
            print(f"Std of rank differences: {np.std(filtered_rank_diffs_gt):.2f}")
            print(f"Min difference: {np.min(filtered_rank_diffs_gt):.2f}")
            print(f"Max difference: {np.max(filtered_rank_diffs_gt):.2f}")
            print(f"Median difference: {np.median(filtered_rank_diffs_gt):.2f}")
            
            # Save filtered results to separate CSV
            filtered_user_rank_diff_gt_df = pd.DataFrame(filtered_user_diffs_gt100)
            filtered_diff_csv_path_gt = f"{output_dir}/{base_name}_user_rank_diff_mingt100.csv"
            filtered_user_rank_diff_gt_df.to_csv(filtered_diff_csv_path_gt, index=False)
            print(f"\n💾 Saved filtered per-user rank differences (min_rank > 100) to: {filtered_diff_csv_path_gt}")
        else:
            print(f"\n⚠️  No users found with min_rank > 100")
        
        # Save all per-user rank differences to CSV
        user_rank_diff_df = pd.DataFrame(user_rank_diffs)
        user_diff_csv_path = f"{output_dir}/{base_name}_user_rank_diff.csv"
        user_rank_diff_df.to_csv(user_diff_csv_path, index=False)
        print(f"\n💾 Saved per-user rank differences (all users) to: {user_diff_csv_path}")
    
    print(f"{'='*80}\n")


def parse_arguments():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="Zeroshot Evaluation with Multiple Rollouts"
    )
    
    # Basic settings
    parser.add_argument("--run_name", type=str, default="zeroshot_eval")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Model checkpoint directory (None for base model)")
    
    # Generation settings
    parser.add_argument("--num_rollouts", type=int, default=1,
                        help="Number of rollouts to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling parameter (-1 for disabled)")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # Prompt settings
    parser.add_argument("--prompt_type", type=str, default="seq_rec",
                        help="Prompt template type")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=128,
                        help="Max words per history item")
    parser.add_argument("--use_brand", action="store_true", default=True,
                        help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", default=True,
                        help="Include category in prompt")
    parser.add_argument("--use_description", action="store_true",
                        help="Include description in prompt")
    parser.add_argument("--use_date", action="store_true", default=True,
                        help="Include purchase date in prompt")
    parser.add_argument("--use_last_item", action="store_true", default=True,
                        help="Emphasize last item in prompt")
    
    # Trigger items settings
    parser.add_argument("--use_trigger_items", action="store_true",
                        help="Use trigger items to emphasize key items in prompt")
    parser.add_argument("--trigger_items_dir", type=str, 
                        default="sasrec_results/trigger_items_from_sequential",
                        help="Directory containing trigger items JSON files")
    parser.add_argument("--trigger_emphasis_text", type=str,
                        default="This item was particularly influential in shaping the user's preferences.",
                        help="Text to add after trigger item to emphasize its importance")
    
    # Evaluation settings
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name for evaluation")
    parser.add_argument("--emb_type", type=str, default="item_meta_only",
                        help="Embedding type")
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=100000)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--use_sentence_transformers", action="store_true",
                        help="Use sentence transformers for evaluation")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for dataset")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Evaluation batch size")
    
    # Split selection
    parser.add_argument("--splits", type=str, nargs="+", default=["test"],
                        help="Dataset splits to evaluate (test, valid, train)")
    
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Seed 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("🎯 Zeroshot Evaluation with Multiple Rollouts")
    print("=" * 80)
    print_arguments(args)
    print(f"\n📋 Evaluating splits: {', '.join(args.splits)}")
    
    # 토크나이저 로드
    print(f"\n📚 Loading tokenizer: {args.model_name}")
    tokenizer = initialize_tokenizer(args.model_name)
    
    # Trigger items 로드 (옵션)
    trigger_items_dict = {}
    if args.use_trigger_items:
        print(f"\n📌 Loading trigger items for all splits...")
        for split in args.splits:
            trigger_items_dict[split] = load_trigger_items(
                args.trigger_items_dir, 
                args.data_name, 
                split=split
            )
    
    # 데이터셋 생성
    print(f"\n📊 Creating datasets...")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        prompt_generator,
        item_metadata,
    ) = create_dataloaders(
        args, 
        tokenizer=tokenizer, 
        apply_chat_template=True,
        trigger_items_train=trigger_items_dict.get("train"),
        trigger_items_valid=trigger_items_dict.get("valid"),
        trigger_items_test=trigger_items_dict.get("test"),
        trigger_emphasis_text=args.trigger_emphasis_text if args.use_trigger_items else None,
    )
    
    # Split별 데이터셋 매핑
    datasets = {
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset,
    }
    
    print(f"✓ Dataset sizes:")
    for split in args.splits:
        print(f"  {split}: {len(datasets[split])} users")
    
    # 모델 경로 설정
    model_path = args.checkpoint_dir if args.checkpoint_dir else args.model_name
    
    # Evaluator 생성 (한 번만)
    print(f"\n🔍 Creating evaluator...")
    evaluator = RecommendationEvaluator(args, model_path)
    
    try:
        # 각 split에 대해 평가 수행
        for split in args.splits:
            print(f"\n{'='*80}")
            print(f"📊 Evaluating {split.upper()} split")
            print(f"{'='*80}")
            
            dataset = datasets[split]
            
            # 프롬프트 및 user_ids 추출
            prompts = []
            user_ids = []
            for i in range(len(dataset)):
                data = dataset[i]
                prompts.append(data['prompt'])
                user_ids.append(data['user_id'])
                if i > args.eval_samples:
                    break
            
            # vLLM 모델 로드 (각 split마다)
            print(f"\n🤖 Loading vLLM model for {split} split...")
            print(f"  Model path: {model_path}")
            
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # Generate rollouts
            generation_results = generate_rollouts_with_vllm(
                llm=llm,
                prompts=prompts,
                user_ids=user_ids,
                num_rollouts=args.num_rollouts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            
            # Generation 완료 후 LLM 객체 삭제하여 메모리 확보
            print(f"\n🧹 Cleaning up LLM to free memory...")
            del llm
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print(f"✓ Memory freed")
            
            # Calculate rollout similarities
            generation_results = calculate_rollout_similarities(
                generation_results=generation_results,
                evaluator=evaluator,
            )
            
            # Evaluate rollouts
            eval_results_df, rank_info = evaluate_rollouts(
                args=args,
                results=generation_results,
                dataset=dataset,
                item_metadata=item_metadata,
                evaluator=evaluator,
                split=split,
            )
            
            # Save results for this split
            save_results(
                generation_results=generation_results,
                eval_results_df=eval_results_df,
                rank_info=rank_info,
                args=args,
                split=split,
                output_dir=args.output_dir,
            )
            
            print(f"\n✅ Completed evaluation for {split} split")
    
    finally:
        # Clean up evaluator
        evaluator.cleanup()
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*80)
    print("✓ All evaluations completed!")
    print("="*80)


if __name__ == "__main__":
    main()
