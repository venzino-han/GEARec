#!/usr/bin/env python3
"""
Rejection Sampling Script for Recommendation System
Generate multiple queries and select best performing ones based on rank improvement
"""

import torch
import argparse
import gc
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from typing import List, Dict, Tuple
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from utils.common import (
    load_json,
    print_arguments,
    get_vllm_model,
    get_sampling_params,
    generate_response_with_vllm,
    initialize_tokenizer,
)
from utils.dataset import create_dataloaders
from evaluator import RecommendationEvaluator

import warnings
warnings.filterwarnings("ignore")


def load_baseline_results(baseline_csv_path: str) -> pd.DataFrame:
    """
    Load baseline results (temp=0.01) from CSV file
    
    Args:
        baseline_csv_path: Path to baseline CSV file
        
    Returns:
        DataFrame with baseline results
    """
    print(f"\n{'='*80}")
    print(f"📄 Loading baseline results from: {baseline_csv_path}")
    print(f"{'='*80}")
    
    df = pd.read_csv(baseline_csv_path)
    
    required_columns = ['user_id', 'rank']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Baseline CSV must contain '{col}' column. Found: {df.columns.tolist()}")
    
    print(f"✓ Loaded baseline results for {len(df)} users")
    print(f"  Average baseline rank: {df['rank'].mean():.2f}")
    print(f"  Median baseline rank: {df['rank'].median():.2f}")
    
    return df


def generate_multiple_queries(
    llm: LLM,
    dataset,
    num_samples: int,
    sampling_params: SamplingParams,
    lora_request: LoRARequest = None,
    batch_size: int = 32,
) -> Dict[int, List[str]]:
    """
    Generate K queries for each user using VLLM sampling
    
    Args:
        llm: VLLM LLM instance
        dataset: Dataset with prompts
        num_samples: Number of queries to generate per user (K)
        sampling_params: VLLM sampling parameters
        lora_request: Optional LoRA request
        batch_size: Batch size for generation
        
    Returns:
        Dictionary mapping user_id to list of generated queries
    """
    print(f"\n{'='*80}")
    print(f"🔄 Generating {num_samples} queries per user")
    print(f"{'='*80}")
    
    # Prepare prompts
    prompts = []
    user_ids = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        prompts.append(data['prompt'])
        user_ids.append(data['user_id'])
    
    # Generate queries with VLLM - all at once
    print(f"Generating {num_samples} samples for {len(prompts)} prompts...")
    print(f"Note: batch_size parameter is ignored, processing all prompts at once")
    
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
    )
    
    # Extract generated texts
    all_generated_texts = []
    for output in tqdm(outputs, desc="Processing outputs"):
        # VLLM returns multiple samples per prompt when n > 1
        sample_texts = [o.text for o in output.outputs]
        all_generated_texts.append(sample_texts)
    
    # Organize by user_id
    user_queries = {}
    for user_id, query_list in zip(user_ids, all_generated_texts):
        if len(query_list) != num_samples:
            print(f"⚠️  Warning: Expected {num_samples} samples but got {len(query_list)} for user {user_id}")
        user_queries[user_id] = query_list
    
    print(f"✓ Generated {num_samples} queries for {len(user_queries)} users")
    
    # Show sample
    sample_user_id = list(user_queries.keys())[0]
    print(f"\n📝 Sample queries for user {sample_user_id}:")
    for idx, query in enumerate(user_queries[sample_user_id][:3]):
        print(f"\n  Query {idx+1}:")
        print(f"  {query[:200]}..." if len(query) > 200 else f"  {query}")
    
    return user_queries


def evaluate_query_batch(
    args,
    evaluator: RecommendationEvaluator,
    generated_texts: List[str],
    dataset,
    split: str = "test",
) -> pd.DataFrame:
    """
    Evaluate a batch of generated queries and return results with ranks
    
    Args:
        args: Arguments
        evaluator: RecommendationEvaluator instance
        generated_texts: List of generated queries
        dataset: Dataset
        split: Dataset split name
        
    Returns:
        DataFrame with evaluation results including ranks
    """
    # Run evaluation
    results = evaluator.evaluate(
        dataset,
        split=split,
        save_log=False,
        pre_generated_texts=generated_texts,
        return_detailed_results=True,  # This should return per-user results
    )
    
    return results


def process_rejection_sampling(
    args,
    user_queries: Dict[int, List[str]],
    baseline_df: pd.DataFrame,
    dataset,
    evaluator: RecommendationEvaluator,
    split: str = "test",
    temp_output_dir: str = "results/rejection_sampling/temp",
) -> pd.DataFrame:
    """
    Process rejection sampling: evaluate all queries and select best ones
    
    Args:
        args: Arguments
        user_queries: Dictionary mapping user_id to list of queries
        baseline_df: DataFrame with baseline results
        dataset: Dataset
        evaluator: RecommendationEvaluator instance
        split: Dataset split name
        temp_output_dir: Temporary directory for CSV files
        
    Returns:
        DataFrame with results for all samples including improvements
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating generated queries")
    print(f"{'='*80}")
    
    # Create temp directory
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Create baseline lookup
    baseline_ranks = baseline_df.set_index('user_id')['rank'].to_dict()
    
    # Prepare all results
    all_results = []
    
    # Evaluate each query for each user
    total_queries = sum(len(queries) for queries in user_queries.values())
    
    print(f"Total queries to evaluate: {total_queries}")
    
    # Get user_ids in dataset order
    user_ids_order = [dataset[i]['user_id'] for i in range(len(dataset))]
    
    # Process each sample index (K samples)
    num_samples = len(next(iter(user_queries.values())))
    
    for sample_idx in range(num_samples):
        print(f"\n🔄 Evaluating sample {sample_idx + 1}/{num_samples}")
        
        # Collect queries for this sample index across all users
        sample_queries = []
        sample_user_ids = []
        
        for user_id in user_ids_order:
            if user_id in user_queries and sample_idx < len(user_queries[user_id]):
                sample_queries.append(user_queries[user_id][sample_idx])
                sample_user_ids.append(user_id)
            else:
                # If missing, use empty string (should not happen)
                sample_queries.append("")
                sample_user_ids.append(user_id)
        
        # Evaluate this batch
        try:
            # Save current run_name
            original_run_name = args.run_name
            
            # Use temporary run name for this evaluation
            args.run_name = f"temp_rs_sample{sample_idx}"
            
            # Run evaluation (this will save CSV automatically)
            evaluator.evaluate(
                dataset,
                split=split,
                save_log=True,  # Enable CSV saving
                pre_generated_texts=sample_queries,
            )
            
            # Restore original run name
            args.run_name = original_run_name
            
            # Find the generated CSV file
            results_dir = Path("results")
            csv_files = list(results_dir.glob(f"temp_rs_sample{sample_idx}_{split}_eval_*.csv"))
            
            if len(csv_files) == 0:
                raise FileNotFoundError(f"Could not find CSV file for sample {sample_idx}")
            
            # Use the most recent CSV file
            csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            
            # Read the CSV
            sample_df = pd.read_csv(csv_file)
            
            # Extract results for each user
            for idx, row in sample_df.iterrows():
                user_id = row['user_id']
                query = row['generated_text']
                current_rank = row['rank']
                baseline_rank = baseline_ranks.get(user_id, np.inf)
                
                # Calculate improvement (positive means better rank)
                improvement = baseline_rank - current_rank
                
                all_results.append({
                    'user_id': user_id,
                    'sample_idx': sample_idx,
                    'generated_query': query,
                    'baseline_rank': baseline_rank,
                    'current_rank': current_rank,
                    'rank_improvement': improvement,
                    'target_item_id': row['target_item_id'],
                    'hit@5': row['hit@5'],
                    'hit@10': row['hit@10'],
                    'hit@20': row['hit@20'],
                })
            
            # Move CSV to temp directory
            temp_csv_path = os.path.join(temp_output_dir, f"sample_{sample_idx}.csv")
            csv_file.rename(temp_csv_path)
            print(f"  Saved evaluation results to: {temp_csv_path}")
        
        except Exception as e:
            print(f"❌ Error evaluating sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Store results with error indicators
            for user_id, query in zip(sample_user_ids, sample_queries):
                baseline_rank = baseline_ranks.get(user_id, np.inf)
                all_results.append({
                    'user_id': user_id,
                    'sample_idx': sample_idx,
                    'generated_query': query,
                    'baseline_rank': baseline_rank,
                    'current_rank': np.inf,
                    'rank_improvement': -np.inf,
                    'target_item_id': -1,
                    'hit@5': 0,
                    'hit@10': 0,
                    'hit@20': 0,
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\n✓ Completed evaluation of {len(results_df)} total samples")
    print(f"  Average rank improvement: {results_df['rank_improvement'].mean():.2f}")
    print(f"  Samples with positive improvement: {(results_df['rank_improvement'] > 0).sum()}")
    
    return results_df


def select_best_samples(
    results_df: pd.DataFrame,
    min_improvement: float = 0,
    selection_strategy: str = "best_improvement",
) -> pd.DataFrame:
    """
    Select best samples based on rank improvement
    
    Args:
        results_df: DataFrame with all results
        min_improvement: Minimum improvement threshold
        selection_strategy: Strategy for selection
            - "best_improvement": Select sample with best improvement per user
            - "best_rank": Select sample with best absolute rank per user
            - "threshold": Select all samples above threshold
            
    Returns:
        DataFrame with selected samples
    """
    print(f"\n{'='*80}")
    print(f"🎯 Selecting best samples (strategy={selection_strategy})")
    print(f"{'='*80}")
    
    if selection_strategy == "best_improvement":
        # For each user, select the sample with best improvement
        selected_df = results_df.loc[results_df.groupby('user_id')['rank_improvement'].idxmax()]
        
    elif selection_strategy == "best_rank":
        # For each user, select the sample with best absolute rank
        selected_df = results_df.loc[results_df.groupby('user_id')['current_rank'].idxmin()]
        
    elif selection_strategy == "threshold":
        # Select all samples above improvement threshold
        selected_df = results_df[results_df['rank_improvement'] >= min_improvement].copy()
        
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    # Filter by minimum improvement
    if min_improvement > 0 and selection_strategy != "threshold":
        before_count = len(selected_df)
        selected_df = selected_df[selected_df['rank_improvement'] >= min_improvement]
        print(f"  Filtered by min_improvement={min_improvement}: {before_count} -> {len(selected_df)} samples")
    
    print(f"\n✓ Selected {len(selected_df)} samples")
    print(f"  Average improvement: {selected_df['rank_improvement'].mean():.2f}")
    print(f"  Average selected rank: {selected_df['current_rank'].mean():.2f}")
    print(f"  Average baseline rank: {selected_df['baseline_rank'].mean():.2f}")
    
    # Show top improvements
    top_improvements = selected_df.nlargest(5, 'rank_improvement')
    print(f"\n📈 Top 5 improvements:")
    for idx, row in top_improvements.iterrows():
        print(f"  User {row['user_id']}: {row['baseline_rank']:.0f} -> {row['current_rank']:.0f} (improvement: {row['rank_improvement']:.0f})")
    
    # Show sample query
    if len(selected_df) > 0:
        print(f"\n📝 Sample selected query:")
        sample_row = selected_df.iloc[0]
        print(f"  User ID: {sample_row['user_id']}")
        print(f"  Baseline rank: {sample_row['baseline_rank']:.0f}")
        print(f"  Selected rank: {sample_row['current_rank']:.0f}")
        print(f"  Improvement: {sample_row['rank_improvement']:.0f}")
        print(f"  Query: {sample_row['generated_query'][:200]}..." if len(sample_row['generated_query']) > 200 else f"  Query: {sample_row['generated_query']}")
    
    return selected_df


def parse_arguments():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="Rejection Sampling for Recommendation System"
    )
    
    # Basic settings
    parser.add_argument("--run_name", type=str, default="rejection_sampling")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 precision")
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint directory for LoRA (optional)")
    parser.add_argument("--enable_lora", action="store_true",
                        help="Enable LoRA adapter")
    parser.add_argument("--lora_adapter_name", type=str, default="lora_adapter",
                        help="LoRA adapter name")
    
    # Generation settings
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of queries to generate per user (K)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for generation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation")
    
    # Evaluation settings
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name for evaluation")
    parser.add_argument("--emb_type", type=str, default="item_meta_only",
                        help="Embedding type")
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_max_tokens", type=int, default=128,
                        help="Maximum tokens for evaluation generation")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Batch size for evaluation")
    parser.add_argument("--eval_samples", type=int, default=100000,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--eval_emb_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--use_sentence_transformers", action="store_true",
                        help="Use sentence transformers for evaluation")
    
    # Dataset settings
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to use")
    parser.add_argument("--prompt_type", type=str, default="seq_rec",
                        help="Prompt template type")
    parser.add_argument("--use_description", action="store_true",
                        help="Include description in prompt")
    parser.add_argument("--use_features", action="store_true",
                        help="Include features in prompt")
    parser.add_argument("--use_brand", action="store_true", default=True,
                        help="Include brand in prompt")
    parser.add_argument("--use_category", action="store_true", default=True,
                        help="Include category in prompt")
    parser.add_argument("--use_date", action="store_true", default=True,
                        help="Include date in prompt")
    parser.add_argument("--use_relative_date", action="store_true",
                        help="Use relative date format")
    parser.add_argument("--use_last_item", action="store_true", default=True,
                        help="Emphasize last item in prompt")
    parser.add_argument("--emphasize_recent_item", action="store_true",
                        help="Emphasize recent purchase item")
    parser.add_argument("--include_target_date", action="store_true",
                        help="Include target date in prompt")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Max history length")
    parser.add_argument("--history_text_max_length", type=int, default=128,
                        help="Max words per history item")
    parser.add_argument("--days_filter", type=int, default=None,
                        help="Filter reviews within N days")
    
    # Baseline settings
    parser.add_argument("--baseline_csv", type=str, required=True,
                        help="Path to baseline CSV (temp=0.01 results)")
    
    # Rejection sampling settings
    parser.add_argument("--min_improvement", type=float, default=0,
                        help="Minimum rank improvement threshold")
    parser.add_argument("--selection_strategy", type=str, default="best_improvement",
                        choices=["best_improvement", "best_rank", "threshold"],
                        help="Strategy for selecting samples")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="results/rejection_sampling",
                        help="Output directory for results")
    parser.add_argument("--save_all_results", action="store_true",
                        help="Save all evaluated samples (not just selected ones)")
    parser.add_argument("--keep_temp_files", action="store_true",
                        help="Keep temporary evaluation CSV files")
    parser.add_argument("--clean_master_logs", action="store_true",
                        help="Clean up temporary master log entries")
    
    args = parser.parse_args()
    
    # Set model_name_dir
    args.model_name_dir = args.model_name.split("/")[-1]
    
    return args


def main():
    """Main rejection sampling function"""
    args = parse_arguments()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("🎲 Rejection Sampling for Recommendation System")
    print("=" * 80)
    print_arguments(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load baseline results
    baseline_df = load_baseline_results(args.baseline_csv)
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer: {args.model_name}")
    tokenizer = initialize_tokenizer(args.model_name)
    
    # Create dataset
    print(f"\n📊 Creating datasets...")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        prompt_generator,
        item_metadata,
    ) = create_dataloaders(args, tokenizer=tokenizer, apply_chat_template=True)
    
    # Select dataset split
    if args.split == "train":
        dataset = train_dataset
    elif args.split == "valid":
        dataset = valid_dataset
    else:
        dataset = test_dataset
    
    print(f"✓ Using {args.split} dataset with {len(dataset)} samples")
    
    # Initialize VLLM model
    print(f"\n🤖 Loading VLLM model: {args.model_name}")
    llm = LLM(
            model=args.model_name, 
            tensor_parallel_size=1,
            max_model_len=args.max_new_tokens + 1024*2,
            max_num_batched_tokens=args.max_new_tokens + 1024*12,
            gpu_memory_utilization = args.gpu_memory_utilization,
            max_num_seqs=64,
            dtype="bfloat16",
        )
    
    # Setup LoRA if needed
    lora_request = None
    if args.enable_lora and args.checkpoint_dir:
        print(f"📦 Loading LoRA adapter from: {args.checkpoint_dir}")
        lora_request = LoRARequest(
            lora_name=args.lora_adapter_name,
            lora_int_id=1,
            lora_path=args.checkpoint_dir,
        )
    
    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=args.num_samples,  # Generate K samples per prompt
    )
    
    print(f"\n🎲 Sampling parameters:")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  n: {args.num_samples}")
    
    # Generate multiple queries
    user_queries = generate_multiple_queries(
        llm=llm,
        dataset=dataset,
        num_samples=args.num_samples,
        sampling_params=sampling_params,
        lora_request=lora_request,
        batch_size=args.batch_size,
    )
    
    # Cleanup VLLM model to free memory
    print("\n🧹 Cleaning up VLLM model...")
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    
    # Initialize evaluator
    print(f"\n📊 Initializing evaluator...")
    # For evaluation, we don't need a checkpoint (using generated texts)
    evaluator = RecommendationEvaluator(args, checkpoint_dir=None, item_metadata=item_metadata)
    
    # Create temp directory for evaluation CSVs
    temp_output_dir = os.path.join(args.output_dir, "temp")
    
    # Process rejection sampling
    results_df = process_rejection_sampling(
        args=args,
        user_queries=user_queries,
        baseline_df=baseline_df,
        dataset=dataset,
        evaluator=evaluator,
        split=args.split,
        temp_output_dir=temp_output_dir,
    )
    
    # Cleanup evaluator
    evaluator.cleanup()
    del evaluator
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save all results if requested
    if args.save_all_results:
        all_results_path = os.path.join(
            args.output_dir,
            f"{args.run_name}_{args.data_name}_{args.split}_all_samples.csv"
        )
        results_df.to_csv(all_results_path, index=False)
        print(f"\n💾 Saved all results to: {all_results_path}")
    
    # Select best samples
    selected_df = select_best_samples(
        results_df=results_df,
        min_improvement=args.min_improvement,
        selection_strategy=args.selection_strategy,
    )
    
    # Save selected samples
    selected_path = os.path.join(
        args.output_dir,
        f"{args.run_name}_{args.data_name}_{args.split}_selected.csv"
    )
    selected_df.to_csv(selected_path, index=False)
    print(f"\n💾 Saved selected samples to: {selected_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"📊 Summary Statistics")
    print(f"{'='*80}")
    print(f"Total samples evaluated: {len(results_df)}")
    print(f"Total samples selected: {len(selected_df)}")
    print(f"Selection rate: {len(selected_df) / len(user_queries) * 100:.1f}%")
    print(f"\nImprovement statistics (selected samples):")
    print(f"  Mean: {selected_df['rank_improvement'].mean():.2f}")
    print(f"  Median: {selected_df['rank_improvement'].median():.2f}")
    print(f"  Min: {selected_df['rank_improvement'].min():.2f}")
    print(f"  Max: {selected_df['rank_improvement'].max():.2f}")
    print(f"\nRank statistics (selected samples):")
    print(f"  Baseline mean: {selected_df['baseline_rank'].mean():.2f}")
    print(f"  Selected mean: {selected_df['current_rank'].mean():.2f}")
    print("=" * 80)
    
    # Cleanup temp files if requested
    if not args.keep_temp_files:
        print(f"\n🧹 Cleaning up temporary files...")
        import shutil
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            print(f"  ✓ Removed: {temp_output_dir}")
        
        # Clean up temp evaluation CSV files
        results_dir = Path("results")
        temp_csv_files = list(results_dir.glob("temp_rs_sample*"))
        for temp_file in temp_csv_files:
            temp_file.unlink()
        if temp_csv_files:
            print(f"  ✓ Removed {len(temp_csv_files)} temporary CSV files")
        
        # Clean up temp log files
        temp_log_files = list(results_dir.glob("temp_rs_sample*.log"))
        for temp_file in temp_log_files:
            temp_file.unlink()
        if temp_log_files:
            print(f"  ✓ Removed {len(temp_log_files)} temporary log files")
    
    print("\n✓ Rejection sampling complete!")


if __name__ == "__main__":
    main()
