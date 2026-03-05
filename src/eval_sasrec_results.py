"""
Evaluate SASRec top-k predictions with retrieval-style metrics.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from utils.reward_function import calculate_ndcg_from_rank, calculate_hit_from_rank


def _load_json(path: Path) -> Dict[str, List[int]]:
    with path.open("r") as f:
        data = json.load(f)
    return data


def _normalize_targets(targets: Dict[str, List[int]]) -> Dict[str, int]:
    normalized = {}
    for key, value in targets.items():
        if isinstance(value, list):
            if not value:
                raise ValueError(f"Empty target list for key={key}")
            normalized[key] = int(value[0])
        else:
            normalized[key] = int(value)
    return normalized


def _normalize_predictions(preds: Dict[str, List[int]]) -> Dict[str, List[int]]:
    normalized = {}
    for key, value in preds.items():
        if not isinstance(value, list):
            raise ValueError(f"Prediction for key={key} must be a list")
        if value and isinstance(value[0], list):
            normalized[key] = [int(v[0]) for v in value]
        else:
            normalized[key] = [int(v) for v in value]
    return normalized


def _load_rank_predictions_from_csv(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    columns = {col.lower(): col for col in df.columns}
    user_col = None
    rank_col = None
    for candidate in ["user_id", "userid", "user"]:
        if candidate in columns:
            user_col = columns[candidate]
            break
    for candidate in ["rank", "target_rank", "gt_rank"]:
        if candidate in columns:
            rank_col = columns[candidate]
            break

    if user_col is None or rank_col is None:
        raise ValueError(
            f"CSV must include user_id and rank columns: {list(df.columns)}"
        )

    ranks: Dict[str, int] = {}
    for user_id, rank in zip(df[user_col], df[rank_col]):
        if pd.isna(user_id) or pd.isna(rank):
            continue
        ranks[str(int(float(user_id)))] = int(float(rank))

    return ranks


def _load_targets_from_sequential(
    data_dir: Path,
    split: str = "test",
) -> Dict[str, int]:
    sequential_file = data_dir / "sequential_data.txt"
    if not sequential_file.exists():
        raise FileNotFoundError(f"Missing {sequential_file}")

    if split == "train":
        target_index = -3
    elif split == "valid":
        target_index = -2
    elif split == "test":
        target_index = -1
    else:
        raise ValueError(f"Invalid split: {split}")

    targets: Dict[str, int] = {}
    with sequential_file.open("r") as f:
        for line in f:
            parts = [int(p) for p in line.strip().split()]
            if not parts:
                continue
            user_id = parts[0]
            target = parts[target_index]
            targets[str(user_id)] = int(target)

    return targets


def _load_item_stats(data_dir: Path) -> Tuple[np.ndarray, set, set]:
    popularity_file = data_dir / "item_popularity.npy"
    novelty_file = data_dir / "item_novelty.npy"
    cold_hot_file = data_dir / "item_cold_hot.json"

    if popularity_file.exists() and novelty_file.exists() and cold_hot_file.exists():
        item_novelty = np.load(novelty_file)
        with cold_hot_file.open("r") as f:
            cold_hot = json.load(f)
        cold_items = set(cold_hot["cold_items"])
        hot_items = set(cold_hot["hot_items"])
        return item_novelty, cold_items, hot_items

    # Fallback: compute from sequential_data.txt
    sequential_file = data_dir / "sequential_data.txt"
    if not sequential_file.exists():
        raise FileNotFoundError(
            f"Missing cached stats and {sequential_file} does not exist"
        )

    max_item_id = 0
    counts = {}
    with sequential_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            items = [int(x) for x in parts[1:-2]]
            for item_id in items:
                counts[item_id] = counts.get(item_id, 0) + 1
                if item_id > max_item_id:
                    max_item_id = item_id

    item_counts = np.zeros(max_item_id + 1, dtype=np.int32)
    for item_id, cnt in counts.items():
        item_counts[item_id] = cnt

    item_popularity = (item_counts + 1) / item_counts.sum()
    item_novelty = -np.log(item_popularity)

    sorted_counts = np.sort(item_counts)[::-1]
    threshold_idx = int(len(sorted_counts) * 0.2)
    threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else 0
    hot_mask = (item_counts >= threshold) & (np.arange(len(item_counts)) > 0)
    cold_mask = (item_counts < threshold) & (np.arange(len(item_counts)) > 0)
    hot_items = set(np.where(hot_mask)[0].tolist())
    cold_items = set(np.where(cold_mask)[0].tolist())

    return item_novelty, cold_items, hot_items


def _compute_metrics(
    targets: Dict[str, int],
    preds: Optional[Dict[str, List[int]]],
    ranks: Optional[Dict[str, int]],
    item_novelty: np.ndarray,
    cold_items: set,
    hot_items: set,
    ks: List[int],
) -> Dict[str, float]:
    pred_keys = set(preds.keys()) if preds is not None else set()
    rank_keys = set(ranks.keys()) if ranks is not None else set()
    shared_keys = sorted(
        set(targets.keys()) & (pred_keys | rank_keys), key=lambda x: int(x)
    )
    if not shared_keys:
        raise ValueError("No shared keys between targets and predictions")

    rank_list = []
    target_items = []
    topk_lists = []
    is_cold_list = []

    for key in shared_keys:
        target = targets[key]
        pred_list = preds.get(key, []) if preds is not None else []
        if ranks is not None and key in ranks:
            rank = int(ranks[key])
        else:
            if not pred_list:
                rank = max(ks) + 1
            else:
                try:
                    rank = pred_list.index(target) + 1
                except ValueError:
                    rank = len(pred_list) + 1

        rank_list.append(rank)
        target_items.append(target)
        topk_lists.append(pred_list)
        is_cold_list.append(target in cold_items)

    ranks_t = torch.tensor(rank_list, dtype=torch.long)
    if ranks_t.min() <= 0:
        ranks_t += 1
    metrics: Dict[str, List[float]] = {f"ndcg@{k}": [] for k in ks}
    metrics.update({f"hit@{k}": [] for k in ks})
    metrics.update({f"cold_ndcg@{k}": [] for k in ks})
    metrics.update({f"cold_hit@{k}": [] for k in ks})
    metrics.update({f"hot_ndcg@{k}": [] for k in ks})
    metrics.update({f"hot_hit@{k}": [] for k in ks})
    metrics.update({f"novelty@{k}": [] for k in ks})
    metrics.update({f"novelty_hit@{k}": [] for k in ks})

    for k in ks:
        ndcg_scores = calculate_ndcg_from_rank(ranks_t, k=k)
        hit_scores = calculate_hit_from_rank(ranks_t, k=k)

        for i, _ in enumerate(target_items):
            if is_cold_list[i]:
                metrics[f"cold_ndcg@{k}"].append(ndcg_scores[i].item())
                metrics[f"cold_hit@{k}"].append(hit_scores[i].item())
            else:
                metrics[f"hot_ndcg@{k}"].append(ndcg_scores[i].item())
                metrics[f"hot_hit@{k}"].append(hit_scores[i].item())

            top_k_items = topk_lists[i][:k]
            if top_k_items:
                avg_novelty = float(
                    np.mean([item_novelty[item] for item in top_k_items])
                )
            else:
                avg_novelty = 0.0
            metrics[f"novelty@{k}"].append(avg_novelty)

            target_novelty = float(item_novelty[target_items[i]])
            novelty_hit = hit_scores[i].item() * target_novelty
            metrics[f"novelty_hit@{k}"].append(novelty_hit)

        metrics[f"ndcg@{k}"].extend(ndcg_scores.cpu().tolist())
        metrics[f"hit@{k}"].extend(hit_scores.cpu().tolist())

    summary = {key: float(np.mean(values)) for key, values in metrics.items()}
    summary["num_samples"] = len(shared_keys)
    return summary


def _format_log(
    metrics: Dict[str, float],
    dataset: str,
    model_name: str,
    ks: List[int],
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    header = [
        "=" * 120,
        f"Master Evaluation Log - {dataset}_sasrec_eval",
        f"Timestamp: {timestamp}",
        f"Dataset: {dataset}",
        f"Model: {model_name}",
        "=" * 120,
        "",
    ]

    metric_cols = ["H@5", "N@5", "H@10", "N@10", "H@20", "N@20"]
    novelty_cols = ["Nov@5", "NH@5", "Nov@10", "NH@10", "Nov@20", "NH@20"]

    def _row(values: List[float]) -> str:
        return "test\t" + "\t".join(f"{v:.4f}" for v in values)

    overall_values = []
    cold_values = []
    hot_values = []
    novelty_values = []
    for k in ks:
        overall_values.extend([metrics[f"hit@{k}"], metrics[f"ndcg@{k}"]])
        cold_values.extend([metrics[f"cold_hit@{k}"], metrics[f"cold_ndcg@{k}"]])
        hot_values.extend([metrics[f"hot_hit@{k}"], metrics[f"hot_ndcg@{k}"]])
        novelty_values.extend([metrics[f"novelty@{k}"], metrics[f"novelty_hit@{k}"]])

    lines = header + [
        "[Overall Metrics]",
        "Split\t" + "\t".join(metric_cols),
        "-" * 70,
        _row(overall_values),
        "",
        "[Cold Items Metrics (80%)]",
        "Split\t" + "\t".join(metric_cols),
        "-" * 70,
        _row(cold_values),
        "",
        "[Hot Items Metrics (Top 20%)]",
        "Split\t" + "\t".join(metric_cols),
        "-" * 70,
        _row(hot_values),
        "",
        "[Novelty Metrics]",
        "Split\t" + "\t".join(novelty_cols),
        "-" * 70,
        _row(novelty_values),
        "-" * 120,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, type=str)
    parser.add_argument("--target_path", type=str, default="")
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="SASRec")
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    data_dir = Path(f"./data/{args.data_name}")

    preds = None
    ranks = None
    if pred_path.suffix.lower() == ".csv":
        ranks = _load_rank_predictions_from_csv(pred_path)
    else:
        preds = _normalize_predictions(_load_json(pred_path))
    if args.target_path:
        target_path = Path(args.target_path)
        targets = _normalize_targets(_load_json(target_path))
    else:
        targets = _load_targets_from_sequential(data_dir, split="test")
    item_novelty, cold_items, hot_items = _load_item_stats(data_dir)

    metrics = _compute_metrics(
        targets=targets,
        preds=preds,
        ranks=ranks,
        item_novelty=item_novelty,
        cold_items=cold_items,
        hot_items=hot_items,
        ks=args.ks,
    )

    log_text = _format_log(metrics, args.data_name, args.model_name, args.ks)
    print(log_text)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".log":
            with output_path.open("w") as f:
                f.write(log_text)
        else:
            with output_path.open("w") as f:
                json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
