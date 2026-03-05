#!/usr/bin/env python3
"""
Similarity-Based Retrieval Evaluation
1. Find the most similar item in user's history to the target item
2. Use that similar item's embedding to retrieve from the full item pool
3. Evaluate retrieval accuracy on train set and save to CSV
"""

import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import torch.nn.functional as F

from utils.common import (
    print_arguments,
    initialize_tokenizer,
    calculate_ndcg,
)
from utils.dataset import create_dataloaders

import warnings
warnings.filterwarnings("ignore")


def load_item_embeddings(args):
    """
    Load or generate item embeddings
    
    Args:
        args: Arguments    
    Returns:
        item_embeddings: Dictionary of {item_id: embedding}
    """
    print(f"\n📊 Loading item embeddings...")
    print(f"  Embedding model: {args.emb_model_name}")
    print(f"  Embedding type: {args.emb_type}")

    item_embeddings = torch.load(f"data_emb/{args.data_name}_{args.emb_type}_{args.emb_model_name.split('/')[-1]}_emb.pt")

    # normalize embeddings
    item_embeddings = F.normalize(item_embeddings, p=2, dim=1)

    return item_embeddings


def load_timestamp_data(args):
    """
    Load timestamp data from user2reviews_with_date.json
    
    Args:
        args: Arguments
    
    Returns:
        user_item_timestamps: Dictionary of {user_id: {item_id: timestamp}}
    """
    print(f"\n📅 Loading timestamp data...")
    
    timestamp_file = f"data/{args.data_name}/user2reviews_with_date.json"
    with open(timestamp_file, 'r') as f:
        user_reviews = json.load(f)
    
    # Create user -> item -> timestamp mapping
    user_item_timestamps = {}
    for user_id_str, reviews in user_reviews.items():
        user_id = int(user_id_str)
        user_item_timestamps[user_id] = {}
        for review in reviews:
            item_id = int(review['item_id'])
            timestamp = review['timestamp']
            user_item_timestamps[user_id][item_id] = timestamp
    
    print(f"  Loaded timestamps for {len(user_item_timestamps)} users")
    return user_item_timestamps


def find_most_similar_history_item(
    target_item_id: int,
    history_item_ids: List[int],
    item_embeddings: torch.Tensor,
    user_id: int = None,
    user_item_timestamps: Dict = None,
    day_filters: int = -1
) -> Tuple[int, float]:
    """
    Find the most similar item in history to the target item
    
    Args:
        target_item_id: Target item ID
        history_item_ids: List of history item IDs
        item_embeddings: Tensor of item embeddings
        user_id: User ID (needed for timestamp filtering)
        user_item_timestamps: Dictionary of {user_id: {item_id: timestamp}}
        day_filters: Number of days to filter (-1 means no filtering)
    
    Returns:
        most_similar_item_id: Most similar item ID from history
        similarity: Cosine similarity score
    """
    
    target_emb = item_embeddings[target_item_id].unsqueeze(0)
    
    # Filter history items by timestamp if day_filters is specified
    filtered_history = history_item_ids
    if day_filters != -1 and user_item_timestamps is not None and user_id is not None:
        if user_id in user_item_timestamps and target_item_id in user_item_timestamps[user_id]:
            target_timestamp = int(user_item_timestamps[user_id][target_item_id])
            day_threshold = day_filters * 24 * 60 * 60  # Convert days to seconds
            
            filtered_history = []
            for item_id in history_item_ids:
                if item_id in user_item_timestamps[user_id]:
                    item_timestamp = int(user_item_timestamps[user_id][item_id])
                    # Only include items within the day filter
                    if target_timestamp - item_timestamp <= day_threshold and item_timestamp < target_timestamp:
                        filtered_history.append(item_id)
            
            # If no items pass the filter, use the last item in history
            if len(filtered_history) == 0:
                filtered_history = [history_item_ids[-1]]
    
    best_similarity = -1.0
    best_item_id = None
    
    for item_id in filtered_history:
        history_emb = item_embeddings[item_id].unsqueeze(0)
        # Cosine similarity (embeddings are already normalized)
        similarity = torch.matmul(target_emb, history_emb.T).squeeze(0)
        
        if similarity > best_similarity:
            best_similarity = similarity.item()
            best_item_id = item_id
    
    return best_item_id, best_similarity


def retrieve_similar_items(
    query_item_id: int,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Retrieve top-k similar items using the query item's embedding
    
    Args:
        query_item_id: Query item ID
        item_embeddings: Tensor of item embeddings
    
    Returns:
        similarities: Tensor of similarity scores
    """
    query_emb = item_embeddings[query_item_id]
    
    # Compute cosine similarities with all items (embeddings are normalized)
    similarities = torch.matmul(item_embeddings, query_emb.unsqueeze(1))

    return similarities


def evaluate_retrieval(
    eval_seq,
    eval_target,
    item_embeddings: torch.Tensor,
    args,
    user_item_timestamps: Dict = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate similarity-based retrieval
    
    Args:
        eval_seq: Dictionary of user IDs and their history
        eval_target: Dictionary of user IDs and their target item ID
        item_embeddings: Tensor of item embeddings
        args: Arguments
        user_item_timestamps: Dictionary of {user_id: {item_id: timestamp}}
    
    Returns:
        results_df: DataFrame with detailed results per user
        metrics: Dictionary of aggregated metrics
    """
    if args.use_last_item:
        print(f"\n🔍 Evaluating last-item-based retrieval...")
    else:
        print(f"\n🔍 Evaluating similarity-based retrieval...")
        if args.day_filters != -1:
            print(f"  📅 Filtering history items within {args.day_filters} days")
    
    results = []
    ranks = []
    
    num_samples = min(len(eval_seq), args.eval_samples)
    
    uids = list(eval_seq.keys())
    for i in tqdm(range(num_samples), desc="Evaluating"):
        user_id = uids[i]
        history = eval_seq[user_id]
        target_item_id = eval_target[user_id]
        
        if args.use_last_item:
            # Use last item in history
            query_item = history[-1]
            similarity_to_target = torch.matmul(
                item_embeddings[target_item_id].unsqueeze(0),
                item_embeddings[query_item].unsqueeze(0).T
            ).squeeze(0).item()
        else:
            # Step 1: Find most similar item in history to target
            query_item, similarity_to_target = find_most_similar_history_item(
                target_item_id,
                history,
                item_embeddings,
                user_id,
                user_item_timestamps,
                args.day_filters
            )

        # Step 2: Retrieve similar items using the query item
        similarities = retrieve_similar_items(
            query_item,
            item_embeddings,
        )
        similarities[history] = -10
        rank = (similarities > similarity_to_target).sum() + 1
        ranks.append(rank.item())
        
        # Store results
        result_dict = {
            'user_id': user_id,
            'target_item_id': target_item_id,
            'query_item': query_item,
            'similarity_to_target': similarity_to_target,
            'rank': rank.item(),
            'history_length': len(history),
        }
        
        # Add descriptive field name based on method
        if args.use_last_item:
            result_dict['last_history_item'] = query_item
        else:
            result_dict['most_similar_history_item'] = query_item
        
        results.append(result_dict)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    ranks = np.array(ranks)
    metrics = {}
    
    # Hit Rate @ K
    for k in [5, 10, 20, 100]:
        hit_rate = np.mean(ranks <= k)
        metrics[f'hit@{k}'] = hit_rate
    
    # NDCG @ K
    for k in [5, 10, 20, 100]:
        sub_ranks = ranks[ranks <= k]
        ndcg = calculate_ndcg(sub_ranks, k)
        metrics[f'ndcg@{k}'] = ndcg
        
    return results_df, metrics


def parse_arguments():
    """Command line arguments"""
    parser = argparse.ArgumentParser(
        description="Similarity-Based Retrieval Evaluation"
    )
    
    # Basic settings
    parser.add_argument("--run_name", type=str, default="similarity_retrieval")
    parser.add_argument("--data_name", type=str, default="beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model settings
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it",
                        help="Model name for tokenizer initialization")
    
    # Embedding settings
    parser.add_argument("--emb_model_name", type=str, 
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model name")
    parser.add_argument("--emb_type", type=str, default="item_meta_only",
                        help="Embedding type")
    parser.add_argument("--emb_batch_size", type=int, default=512,
                        help="Batch size for embedding generation")
    parser.add_argument("--max_history_len", type=int, default=8,
                        help="Maximum history length")
    
    # Evaluation settings
    parser.add_argument("--eval_split", type=str, default="train",
                        choices=["train", "valid", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--eval_samples", type=int, default=100000,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--use_last_item", action="store_true",
                        help="Use last item in history instead of most similar item")
    parser.add_argument("--day_filters", type=int, default=-1,
                        help="Filter history items within N days (-1 means no filtering)")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("🔍 Similarity-Based Retrieval Evaluation")
    print("=" * 80)
    print_arguments(args)
    
    # Load tokenizer
    print(f"\n📚 Loading tokenizer: {args.model_name}")
    tokenizer = initialize_tokenizer(args.model_name)
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    if args.eval_split == "train":
        target_index = -3
    elif args.eval_split == "valid":
        target_index = -2
    elif args.eval_split == "test":
        target_index = -1
    else:
        raise ValueError(f"Unknown eval_split: {args.eval_split}")

    
    eval_seq = {}
    eval_target = {}

    with open(f"data/{args.data_name}/sequential_data.txt", "r") as f:
        sequential_data = f.readlines()
    for line in sequential_data:
        parts = line.strip().split()
        parts = [int(p) for p in parts]
        user_id = parts[0]
        history = parts[1:target_index]
        target = parts[target_index]
        eval_seq[user_id] = history[-args.max_history_len:]
        eval_target[user_id] = target
    
    # Select dataset split
    
    print(f"  Evaluating on {args.eval_split} set: {len(eval_seq)} samples")
    
    # Load item embeddings
    item_embeddings = load_item_embeddings(args)
    
    # Load timestamp data if day_filters is specified
    user_item_timestamps = None
    if args.day_filters != -1:
        user_item_timestamps = load_timestamp_data(args)
    
    # Evaluate retrieval
    results_df, metrics = evaluate_retrieval(
        eval_seq,
        eval_target,
        item_embeddings,
        args,
        user_item_timestamps
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("📊 Evaluation Results")
    print("=" * 80)
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")
    print("=" * 80)
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine method prefix for filenames
    method_prefix = "last_item_retrieval" if args.use_last_item else args.run_name
    
    # Add day filter suffix if specified
    day_suffix = f"_day{args.day_filters}" if args.day_filters != -1 else ""
    
    # Save detailed results
    csv_filename = f"{method_prefix}_{args.data_name}_{args.eval_split}{day_suffix}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved detailed results to: {csv_path}")
    
    # Save metrics
    metrics_filename = f"{method_prefix}_{args.data_name}_{args.eval_split}{day_suffix}_metrics_{timestamp}.json"
    metrics_path = os.path.join(args.output_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to: {metrics_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
