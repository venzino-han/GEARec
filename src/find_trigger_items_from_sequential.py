"""
Find trigger items from sequential_data.txt file, or compute item-based retrieval rank.

Modes:
  trigger: Find the past history item most similar to the user embedding (existing behavior)
  rank:    Use the mean embedding of the user's recent N history items as a query,
           retrieve over all items, and record the rank of the ground-truth target item.

File format (sequential_data.txt):
  user_id item1 item2 ... itemN
  
Where:
  - First number: user_id
  - Rest: sequential purchase history
  - Index -3: train target
  - Index -2: val target
  - Index -1: test target
"""

import csv
import torch
import json
import os
from typing import Dict, List, Optional, Tuple
import argparse
import math


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.
    
    Args:
        a: Tensor of shape (d,) or (n, d)
        b: Tensor of shape (d,) or (m, d)
    
    Returns:
        Similarity score(s)
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    
    # Normalize
    # a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    # b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute similarity
    similarity = torch.mm(a, b.t())
    return similarity


def load_sequential_data(filepath: str) -> Dict[int, List[int]]:
    """
    Load sequential data from file.
    
    Args:
        filepath: Path to sequential_data.txt
    
    Returns:
        Dict mapping user_id to list of item_ids
    """
    user_sequences = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 2:
                continue
            
            user_id = parts[0]
            items = parts[1:]
            user_sequences[user_id] = items
    
    return user_sequences


def get_split_items(items: List[int], split: str) -> Tuple[List[int], int]:
    """
    Get past items and target for a specific split.
    
    Args:
        items: Full sequence of items
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (past_items, target_item)
        - For train: items before index -3, target at -3
        - For val: items before index -2, target at -2
        - For test: items before index -1, target at -1
    """
    if len(items) < 3:
        # Not enough items for all splits
        return [], -1
    
    if split == 'train':
        # Items before train target (index -3)
        past_items = items[:-3]
        target = items[-3]
    elif split == 'val' or split == 'valid':
        # Items before val target (index -2)
        past_items = items[:-2]
        target = items[-2]
    elif split == 'test':
        # Items before test target (index -1)
        past_items = items[:-1]
        target = items[-1]
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return past_items, target


def find_trigger_items_for_split(
    user_sequences: Dict[int, List[int]],
    item_embeddings: torch.Tensor,
    split: str,
    top_n: int = 5,
    device: str = "cpu",
) -> List[Dict]:
    """
    Find trigger items for all users in a specific split and compute the rank of
    the ground-truth target item.

    Trigger item: among the most recent top_n history items, the one most similar
                  to the target item embedding.
    Rank: rank of target item when using the trigger item's embedding as query
          against all items (past interacted items masked out).

    Args:
        user_sequences: Dict mapping user_id to full item sequence
        item_embeddings: Item embeddings tensor (num_items, emb_dim)
        split: 'train', 'val', or 'test'
        top_n: Number of most-recent history items to search trigger from
        device: Torch device for computation

    Returns:
        List of dicts with keys: user_id, target_item_id, rank, best_rank_item_id,
                                  trigger_item_id, query_items, num_query_items
    """
    results = []
    skipped = 0
    num_items = item_embeddings.shape[0]

    item_emb = item_embeddings.to(device).float()

    print(f"Processing {len(user_sequences)} users for {split} split (top_n={top_n})...")

    for idx, (user_id, full_items) in enumerate(user_sequences.items()):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(user_sequences)} users...")

        # Get past items for this split
        past_items, target = get_split_items(full_items, split)

        if len(past_items) == 0 or target == -1:
            skipped += 1
            continue

        if target < 0 or target >= num_items:
            skipped += 1
            continue

        # Most recent top_n valid past items
        recent_ids = [i for i in past_items[-top_n:] if 0 <= i < num_items]
        if not recent_ids:
            skipped += 1
            continue

        # Find trigger: among recent_ids, the one most similar to the target item
        target_emb = item_emb[target]                      # (d,)
        recent_embs = item_emb[recent_ids]                 # (k, d)
        target_sims = recent_embs @ target_emb             # (k,)
        trigger_item_id = recent_ids[target_sims.argmax().item()]

        # Compute rank: use trigger item embedding as query over all items
        query_emb = item_emb[trigger_item_id]              # (d,)
        all_sims = item_emb @ query_emb                    # (num_items,)
        interacted = [i for i in past_items if 0 <= i < num_items]
        if interacted:
            all_sims[interacted] = float("-inf")
        target_sim = all_sims[target].item()
        rank = int((all_sims > target_sim).sum().item()) + 1
        best_item = int(all_sims.argmax().item())

        results.append({
            "user_id": user_id,
            "target_item_id": target,
            "rank": rank,
            "best_rank_item_id": best_item,
            "trigger_item_id": trigger_item_id,
            "query_items": ",".join(map(str, recent_ids)),
            "num_query_items": len(recent_ids),
        })
    
    print(f"  Completed! Found trigger items for {len(results)} users.")
    print(f"  Skipped {skipped} users (not enough items or invalid indices)")
    
    return results


def load_user_reviews_with_date(data_dir: str, dataset_name: str) -> Dict[int, List[Dict]]:
    """Load user2reviews_with_date.json → {user_id: [{item_id, timestamp, ...}, ...]}"""
    path = f"{data_dir}/{dataset_name}/user2reviews_with_date.json"
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def build_timestamp_indices(
    user_reviews: Dict[int, List[Dict]],
    num_items: int,
    split: str,
) -> Tuple[Dict[int, int], torch.Tensor]:
    """
    Build per-user target timestamps and a per-item max-timestamp tensor.

    Args:
        user_reviews: {user_id: [{item_id, timestamp, ...}, ...]}
        num_items: Total number of items (size of item embedding matrix)
        split: 'train', 'val'/'valid', or 'test'

    Returns:
        user_target_ts: {user_id -> target item's purchase timestamp}
        item_max_ts:    LongTensor of shape (num_items,) — max timestamp seen
                        for each item across all users (-1 if never seen)
    """
    split_index = {"train": -3, "val": -2, "valid": -2, "test": -1}
    idx = split_index[split]

    user_target_ts: Dict[int, int] = {}
    item_max_ts = torch.full((num_items,), -1, dtype=torch.long)

    for user_id, reviews in user_reviews.items():
        try:
            target_review = reviews[idx]
        except IndexError:
            continue
        user_target_ts[user_id] = int(target_review["timestamp"])

        for review in reviews:
            item_id = int(review["item_id"])
            ts = int(review["timestamp"])
            if 0 <= item_id < num_items and ts > item_max_ts[item_id].item():
                item_max_ts[item_id] = ts

    return user_target_ts, item_max_ts


def find_retrieval_rank_for_split(
    user_sequences: Dict[int, List[int]],
    item_embeddings: torch.Tensor,
    split: str,
    top_n: int = 5,
    device: str = "cpu",
    candidate_days: int = -1,
    user_target_ts: Optional[Dict[int, int]] = None,
    item_max_ts: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """
    For each user, average the embeddings of the most recent top_n history items
    and use that as a query against all item embeddings.  Record the 1-indexed rank
    of the ground-truth target item in the sorted result list.

    Args:
        user_sequences: Dict mapping user_id to full item sequence
        item_embeddings: Item embeddings tensor (num_items, emb_dim)
        split: 'train', 'val', or 'test'
        top_n: Number of most-recent history items to use as query
        device: 'cpu' or 'cuda'
        candidate_days: If > 0, restrict retrieval candidates to items whose most-recent
                        interaction (across all users) falls within this many days before
                        the user's target purchase timestamp.  Requires user_target_ts and
                        item_max_ts.  -1 disables filtering.
        user_target_ts: {user_id -> target item's purchase timestamp (Unix seconds)}
        item_max_ts: LongTensor (num_items,) — max interaction timestamp per item

    Returns:
        List of dicts with keys: user_id, target_item_id, rank, best_rank_item_id,
                                 query_items, num_query_items
    """
    results = []
    skipped = 0
    num_items = item_embeddings.shape[0]

    item_emb = item_embeddings.to(device).float()
    use_date_filter = (
        candidate_days > 0
        and user_target_ts is not None
        and item_max_ts is not None
    )
    item_max_ts_dev = item_max_ts.to(device) if use_date_filter else None

    day_label = f", candidate_days={candidate_days}" if use_date_filter else ""
    print(f"Processing {len(user_sequences)} users for {split} split (top_n={top_n}{day_label})...")

    for idx, (user_id, full_items) in enumerate(user_sequences.items()):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(user_sequences)} users...")

        past_items, target = get_split_items(full_items, split)

        if len(past_items) == 0 or target == -1:
            skipped += 1
            continue

        if target < 0 or target >= num_items:
            skipped += 1
            continue

        # Take the most recent top_n items
        recent_items = past_items[-top_n:]
        valid_ids = [i for i in recent_items if 0 <= i < num_items]

        if not valid_ids:
            skipped += 1
            continue

        # Mean pooling of query item embeddings
        query_emb = item_emb[valid_ids].mean(dim=0)  # (d,)

        # Dot-product similarity against all items  (num_items,)
        sims = item_emb @ query_emb

        # Mask out all items the user has already interacted with (past_items)
        interacted = [i for i in past_items if 0 <= i < num_items]
        if interacted:
            sims[interacted] = float("-inf")

        # Optionally restrict candidates to items active within candidate_days
        if use_date_filter:
            target_ts = user_target_ts.get(user_id, -1)
            if target_ts > 0:
                window_start = target_ts - candidate_days * 86400
                outside_window = (item_max_ts_dev < window_start) | (item_max_ts_dev < 0)
                outside_window[target] = False  # always keep ground-truth as candidate
                sims[outside_window] = float("-inf")

        target_sim = sims[target].item()
        # rank = number of items with strictly higher similarity + 1
        rank = int((sims > target_sim).sum().item()) + 1
        best_item = int(sims.argmax().item())

        results.append({
            "user_id": user_id,
            "target_item_id": target,
            "rank": rank,
            "best_rank_item_id": best_item,
            "query_items": ",".join(map(str, valid_ids)),
            "num_query_items": len(valid_ids),
        })

    print(f"  Completed! Computed ranks for {len(results)} users.")
    print(f"  Skipped {skipped} users (not enough items or invalid indices)")

    return results


def summarize_ranks(results: List[Dict], ks: List[int] = [1, 5, 10, 20, 50]) -> Dict:
    """Compute HR@K and NDCG@K from a list of rank dicts."""
    metrics: Dict = {}
    ranks = [r["rank"] for r in results]
    for k in ks:
        hits = sum(1 for rk in ranks if rk <= k)
        hr = hits / len(ranks) if ranks else 0.0
        ndcg = sum(1.0 / math.log2(rk + 1) for rk in ranks if rk <= k) / len(ranks) if ranks else 0.0
        metrics[f"HR@{k}"] = hr
        metrics[f"NDCG@{k}"] = ndcg
    metrics["MRR"] = sum(1.0 / rk for rk in ranks) / len(ranks) if ranks else 0.0
    metrics["num_users"] = len(ranks)
    return metrics


def process_dataset_rank(
    dataset_name: str,
    data_dir: str = "data",
    sasrec_results_dir: str = "sasrec_results",
    output_dir: str = "sasrec_results/retrieval_rank",
    top_n: int = 5,
    splits: List[str] = None,
    device: str = "cpu",
    candidate_days: int = -1,
):
    """
    For a dataset, compute the item-based retrieval rank for each user/split and
    save results to CSV files.

    Args:
        dataset_name: Name of the dataset (e.g., 'toys', 'beauty', 'sports', 'yelp')
        data_dir: Directory containing data folders
        sasrec_results_dir: Directory containing SASRec embeddings
        output_dir: Directory to save CSV result files
        top_n: Number of recent history items to use as query
        splits: List of splits to process (default: ['train', 'val', 'test'])
        device: Torch device to use for computation
        candidate_days: If > 0, restrict retrieval pool to items interacted within
                        this many days before each user's target purchase. -1 disables.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    day_label = f", candidate_days={candidate_days}" if candidate_days > 0 else ""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}  [mode=rank, top_n={top_n}{day_label}]")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    seq_data_path = f"{data_dir}/{dataset_name}/sequential_data.txt"
    if not os.path.exists(seq_data_path):
        print(f"Error: Sequential data not found at {seq_data_path}")
        return

    print(f"\nLoading sequential data from {seq_data_path}...")
    user_sequences = load_sequential_data(seq_data_path)
    print(f"  Loaded sequences for {len(user_sequences)} users")

    item_emb_path = f"{sasrec_results_dir}/SASRec_{dataset_name}_item_embeddings.pt"
    if not os.path.exists(item_emb_path):
        print(f"Error: Item embeddings not found at {item_emb_path}")
        return

    print(f"\nLoading item embeddings from {item_emb_path}...")
    item_embeddings = torch.load(item_emb_path, map_location="cpu")
    print(f"  Item embeddings shape: {item_embeddings.shape}")

    # Load timestamps once if date filtering is requested
    user_reviews = None
    if candidate_days > 0:
        reviews_path = f"{data_dir}/{dataset_name}/user2reviews_with_date.json"
        if not os.path.exists(reviews_path):
            print(f"Warning: {reviews_path} not found — candidate_days filtering disabled.")
            candidate_days = -1
        else:
            print(f"\nLoading review timestamps from {reviews_path}...")
            user_reviews = load_user_reviews_with_date(data_dir, dataset_name)
            print(f"  Loaded reviews for {len(user_reviews)} users")

    day_suffix = f"_days{candidate_days}" if candidate_days > 0 else ""

    for split in splits:
        print(f"\n--- Processing {split} split ---")

        user_target_ts = None
        item_max_ts = None
        if candidate_days > 0 and user_reviews is not None:
            print(f"  Building timestamp indices for {split} split...")
            user_target_ts, item_max_ts = build_timestamp_indices(
                user_reviews, item_embeddings.shape[0], split
            )
            print(f"  Target timestamps available for {len(user_target_ts)} users")

        results = find_retrieval_rank_for_split(
            user_sequences,
            item_embeddings,
            split,
            top_n=top_n,
            device=device,
            candidate_days=candidate_days,
            user_target_ts=user_target_ts,
            item_max_ts=item_max_ts,
        )

        if not results:
            print(f"  No results for {split} split, skipping save.")
            continue

        output_path = f"{output_dir}/SASRec_{dataset_name}_{split}_top{top_n}_retrieval_rank.csv"
        fieldnames = ["user_id", "target_item_id", "rank", "best_rank_item_id", "query_items", "num_query_items"]
        print(f"  Saving results to {output_path}...")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        summary = summarize_ranks(results)
        print(f"  Summary ({split}, top_n={top_n}{day_label}):")
        for metric, val in summary.items():
            if metric == "num_users":
                print(f"    {metric}: {val}")
            else:
                print(f"    {metric}: {val:.4f}")

        print(f"  Successfully saved {len(results)} rows!")


def process_dataset(
    dataset_name: str,
    data_dir: str = "data",
    sasrec_results_dir: str = "sasrec_results",
    output_dir: str = "sasrec_results/trigger_items_from_sequential",
    top_n: int = 5,
    device: str = "cpu",
):
    """
    Process a single dataset to find trigger items from sequential_data.txt.

    Args:
        dataset_name: Name of the dataset (e.g., 'toys', 'beauty', 'sports', 'yelp')
        data_dir: Directory containing data folders
        sasrec_results_dir: Directory containing SASRec embeddings
        output_dir: Directory to save trigger items CSV files
        top_n: Number of most-recent history items to use as rank query
        device: Torch device for computation
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequential data
    seq_data_path = f"{data_dir}/{dataset_name}/sequential_data.txt"
    if not os.path.exists(seq_data_path):
        print(f"Error: Sequential data not found at {seq_data_path}")
        return
    
    print(f"\nLoading sequential data from {seq_data_path}...")
    user_sequences = load_sequential_data(seq_data_path)
    print(f"  Loaded sequences for {len(user_sequences)} users")
    
    # Load item embeddings (shared across splits)
    item_emb_path = f"{sasrec_results_dir}/SASRec_{dataset_name}_item_embeddings.pt"
    if not os.path.exists(item_emb_path):
        print(f"Error: Item embeddings not found at {item_emb_path}")
        return
    
    print(f"\nLoading item embeddings from {item_emb_path}...")
    item_embeddings = torch.load(item_emb_path)
    print(f"  Item embeddings shape: {item_embeddings.shape}")
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n--- Processing {split} split ---")
        
        # Find trigger items and compute target rank
        results = find_trigger_items_for_split(
            user_sequences,
            item_embeddings,
            split,
            top_n=top_n,
            device=device,
        )

        if not results:
            print(f"  No results for {split} split, skipping save.")
            continue

        # Save results as CSV
        output_path = f"{output_dir}/SASRec_{dataset_name}_{split}_trigger_items.csv"
        fieldnames = ["user_id", "target_item_id", "rank", "best_rank_item_id", "trigger_item_id", "query_items", "num_query_items"]
        print(f"  Saving trigger items to {output_path}...")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        summary = summarize_ranks(results)
        print(f"  Summary ({split}):")
        for metric, val in summary.items():
            if metric == "num_users":
                print(f"    {metric}: {val}")
            else:
                print(f"    {metric}: {val:.4f}")

        print(f"  Successfully saved {len(results)} trigger items!")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find trigger items (mode=trigger) or compute item-based retrieval rank "
            "(mode=rank) from sequential_data.txt"
        )
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='trigger',
        choices=['trigger', 'rank'],
        help=(
            "trigger: among recent top_n items, find the one most similar to target, "
            "then use it as query to record target rank in full retrieval. "
            "rank: use mean of recent top_n item embeddings as query, record target rank. "
            "(default: trigger)"
        )
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['toys', 'beauty', 'sports', 'yelp'],
        help='List of datasets to process (default: toys beauty sports yelp)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to process for rank mode (default: train val test)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing data folders (default: data)'
    )
    parser.add_argument(
        '--sasrec_results_dir',
        type=str,
        default='sasrec_results',
        help='Directory containing SASRec embeddings (default: sasrec_results)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help=(
            'Directory to save output files. '
            'Defaults to sasrec_results/trigger_items_from_sequential (trigger mode) '
            'or sasrec_results/retrieval_rank (rank mode).'
        )
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=5,
        help='Number of most-recent history items to use as rank query (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='[rank mode] Device for tensor computation (default: cuda if available else cpu)'
    )
    parser.add_argument(
        '--candidate_days',
        type=int,
        default=-1,
        help=(
            '[rank mode] Restrict retrieval candidate pool to items whose most-recent '
            'interaction (across all users) is within this many days before each user\'s '
            'target purchase timestamp.  Requires user2reviews_with_date.json. '
            '-1 disables filtering (default: -1)'
        )
    )

    args = parser.parse_args()

    # Resolve default output_dir per mode
    if args.output_dir is None:
        if args.mode == 'trigger':
            args.output_dir = 'sasrec_results/trigger_items_from_sequential'
        else:
            args.output_dir = 'sasrec_results/retrieval_rank'

    print("="*60)
    if args.mode == 'trigger':
        print("Finding Trigger Items from Sequential Data")
    else:
        print("Computing Item-Based Retrieval Rank")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Datasets: {args.datasets}")
    print(f"Data directory: {args.data_dir}")
    print(f"SASRec results directory: {args.sasrec_results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Top-N history items: {args.top_n}")
    if args.mode == 'rank':
        print(f"Splits: {args.splits}")
        if args.candidate_days > 0:
            print(f"Candidate days filter: {args.candidate_days}")
    print("\nSplit definitions:")
    print("  - train: items before index -3 (train target at -3)")
    print("  - val:   items before index -2 (val target at -2)")
    print("  - test:  items before index -1 (test target at -1)")

    for dataset in args.datasets:
        try:
            if args.mode == 'trigger':
                process_dataset(
                    dataset_name=dataset,
                    data_dir=args.data_dir,
                    sasrec_results_dir=args.sasrec_results_dir,
                    output_dir=args.output_dir,
                    top_n=args.top_n,
                    device=args.device,
                )
            else:
                process_dataset_rank(
                    dataset_name=dataset,
                    data_dir=args.data_dir,
                    sasrec_results_dir=args.sasrec_results_dir,
                    output_dir=args.output_dir,
                    top_n=args.top_n,
                    splits=args.splits,
                    device=args.device,
                    candidate_days=args.candidate_days,
                )
        except Exception as e:
            print(f"\nError processing dataset {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == "__main__":
    main()
