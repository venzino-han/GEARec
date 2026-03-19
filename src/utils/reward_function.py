"""
Reward Functions for RL4Rec
NDCG-based reward calculation and TRL integration
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import ray
from pathlib import Path
import argparse
import nltk
from nltk.corpus import stopwords
import re


def extract_query_from_tags(text: str, tag: str = "query") -> str:
    """
    Extract the content inside a specific XML-like tag from text.
    Returns the original text if the tag is not found.

    Args:
        text: input text
        tag: name of the tag to extract (default: "query")

    Returns:
        text inside the tag, or the original text if the tag is absent
        
    Example:
        >>> text = "<thinking>...</thinking><query>camping gear</query>"
        >>> extract_query_from_tags(text)
        "camping gear"
    """
    # Extract tag content using regex (case-insensitive, include newlines)
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Extract tag content and strip surrounding whitespace
        return match.group(1).strip()
    else:
        # Return original text if tag is not found
        return text


def calculate_dcg(relevance_scores: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Compute DCG (Discounted Cumulative Gain).

    Args:
        relevance_scores: [batch_size, num_items] relevance scores
        k: compute up to top-k only (None means all)

    Returns:
        DCG scores [batch_size]
    """
    if k is not None:
        relevance_scores = relevance_scores[:, :k]
    
    # DCG = sum(rel_i / log2(i + 2)) for i in range(k)
    positions = torch.arange(1, relevance_scores.shape[1] + 1, device=relevance_scores.device)
    discounts = torch.log2(positions + 1.0)
    dcg = (relevance_scores / discounts).sum(dim=1)
    
    return dcg


def calculate_ndcg_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute NDCG from rank.

    Args:
        ranks: [batch_size] rank of the GT item (1-indexed)
        k: Top-K NDCG

    Returns:
        NDCG scores [batch_size]
    """
    # NDCG = 0 when rank > k
    ndcg_scores = torch.zeros_like(ranks, dtype=torch.float32)

    # Compute only for ranks within k
    valid_mask = ranks <= k
    valid_ranks = ranks[valid_mask]
    
    # DCG = 1 / log2(rank + 1)
    dcg = 1.0 / torch.log2(valid_ranks.float() + 1.0)
    
    # IDCG = 1 / log2(2) (ideal case, rank=1)
    idcg = 1.0 / torch.log2(torch.tensor(2.0, device=ranks.device))
    
    # NDCG = DCG / IDCG
    ndcg_scores[valid_mask] = dcg / idcg
    
    return ndcg_scores


def calculate_hit_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute Hit@K from rank.

    Args:
        ranks: [batch_size] rank of the GT item (1-indexed)
        k: Top-K

    Returns:
        Hit scores [batch_size]
    """
    # 1 if rank is within k, otherwise 0
    hit_scores = (ranks <= k).float()
    return hit_scores


def calculate_mrr_from_rank(ranks: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Compute MRR (Mean Reciprocal Rank) from rank.

    Args:
        ranks: [batch_size] rank of the GT item (1-indexed)
        k: Top-K

    Returns:
        MRR scores [batch_size]
    """
    # MRR = 0 when rank > k
    mrr_scores = torch.zeros_like(ranks, dtype=torch.float32)

    # Compute only for ranks within k
    valid_mask = ranks <= k
    valid_ranks = ranks[valid_mask]
    
    # MRR = 1 / rank
    mrr_scores[valid_mask] = 1.0 / valid_ranks.float()
    
    return mrr_scores


def calculate_novelty_ndcg(
    ranks: torch.Tensor, 
    item_weights: torch.Tensor,
    k: int = 10, 
) -> torch.Tensor:
    """
    Popularity-based novelty score calculation
    
    Compute novelty by multiplying NDCG by the inverse popularity (item_weights).
    Higher novelty is assigned when unpopular items are predicted with higher rank.

    Args:
        ranks: [batch_size] rank of the GT item (1-indexed)
        item_weights: [batch_size] popularity weight (inverse) of each target item
        k: Top-K
        target_rank: (unused, kept for backward compatibility)
    
    Returns:
        novelty scores [batch_size] = NDCG × item_weights
    """
    # 1. Compute NDCG from ranks
    ndcg_scores = calculate_ndcg_from_rank(ranks, k=k)

    # 2. Multiply NDCG by popularity weight
    # Correctly predicting unpopular items (high weight) yields high novelty
    novelty = ndcg_scores * item_weights
    
    return novelty


def calculate_ndcg(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain).

    Args:
        predicted_scores: [batch_size, num_items] or [batch_size, 1+num_negs] predicted scores
        target_items: [batch_size] list of actual target item IDs
        history_items: [batch_size, *] per-user history item ID lists
        k: Top-K NDCG (default: 10)
        use_negatives_only: when True, use only target+negatives (scores shape [batch_size, 1+num_negs])
    
    Returns:
        NDCG scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    ndcg_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Consider only target + negatives
        # scores shape: [batch_size, 1 + num_negs]
        # target is always at index 0
        for i in range(batch_size):
            scores = predicted_scores[i]  # [1 + num_negs]
            k_actual = min(k, len(scores))
            
            # Extract Top-K
            top_k_scores, top_k_indices = torch.topk(scores, k=k_actual)

            # Check whether target (index 0) is in top-k
            relevance = torch.zeros(k_actual, device=predicted_scores.device)
            target_positions = (top_k_indices == 0).nonzero(as_tuple=True)[0]
            
            if len(target_positions) > 0:
                position = target_positions[0].item()
                relevance[position] = 1.0
                
                # Compute DCG
                dcg = calculate_dcg(relevance.unsqueeze(0), k=k_actual)[0]

                # Compute IDCG
                ideal_relevance = torch.zeros(k_actual, device=predicted_scores.device)
                ideal_relevance[0] = 1.0
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=k_actual)[0]
                
                ndcg_scores[i] = dcg / (idcg + 1e-10)
            else:
                ndcg_scores[i] = 0.0
    else:
        # Consider all items (original logic)
        for i in range(batch_size):
            # 1. Mask out history items
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
                history_mask[history_items[i]] = True
                history_mask[target_items[i]] = False
                scores[history_mask] = -float('inf')
            
            # 2. Extract Top-K items
            top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))

            # 3. Compute relevance (1 at position of target item in top-k, else 0)
            relevance = torch.zeros(k, device=predicted_scores.device)
            target_item = target_items[i]
            
            # Find position of target item in top-k
            target_positions = (top_k_indices == target_item).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                position = target_positions[0].item()
                relevance[position] = 1.0
            
            # 4. Compute DCG
            if relevance.sum() > 0:
                dcg = calculate_dcg(relevance.unsqueeze(0), k=k)[0]

                # 5. Compute IDCG (Ideal DCG) - ideal case (target at rank 1)
                ideal_relevance = torch.zeros(k, device=predicted_scores.device)
                ideal_relevance[0] = 1.0
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=k)[0]
                
                # 6. NDCG = DCG / IDCG
                ndcg_scores[i] = dcg / (idcg + 1e-10)
            else:
                # NDCG = 0 when target is not in top-k
                ndcg_scores[i] = 0.0
    
    return ndcg_scores


def calculate_hit_rate(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    Compute Hit@K (1 if target is in top-k, else 0).

    Args:
        predicted_scores: [batch_size, num_items] or [batch_size, 1+num_negs]
        target_items: [batch_size]
        history_items: [batch_size, *]
        k: Top-K
        use_negatives_only: when True, use only target+negatives
    
    Returns:
        Hit scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    hit_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Consider only target + negatives
        for i in range(batch_size):
            scores = predicted_scores[i]
            k_actual = min(k, len(scores))

            # Extract Top-K
            _, top_k_indices = torch.topk(scores, k=k_actual)

            # Check whether target (index 0) is in top-k
            if 0 in top_k_indices:
                hit_scores[i] = 1.0
    else:
        # Consider all items (original logic)
        for i in range(batch_size):
            # Mask out history items
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool, device=scores.device)
                history_mask[history_items[i]] = True
                history_mask[target_items[i]] = False
                scores[history_mask] = -float('inf')
            
            # Extract Top-K
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))

            # Check whether target is in top-k
            if target_items[i] in top_k_indices:
                hit_scores[i] = 1.0
    
    return hit_scores


def calculate_mrr(
    predicted_scores: torch.Tensor,
    target_items: List[int],
    history_items: List[List[int]],
    k: int = 10,
    use_negatives_only: bool = False,
) -> torch.Tensor:
    """
    Compute MRR (Mean Reciprocal Rank).

    Args:
        predicted_scores: [batch_size, num_items] or [batch_size, 1+num_negs]
        target_items: [batch_size]
        history_items: [batch_size, *]
        k: Top-K
        use_negatives_only: when True, use only target+negatives
    
    Returns:
        MRR scores [batch_size]
    """
    batch_size = predicted_scores.shape[0]
    mrr_scores = torch.zeros(batch_size, device=predicted_scores.device)
    
    if use_negatives_only:
        # Consider only target + negatives
        for i in range(batch_size):
            scores = predicted_scores[i]
            k_actual = min(k, len(scores))

            # Extract Top-K
            _, top_k_indices = torch.topk(scores, k=k_actual)

            # Find rank of target (index 0)
            target_positions = (top_k_indices == 0).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                rank = target_positions[0].item() + 1  # 1-indexed rank
                mrr_scores[i] = 1.0 / rank
    else:
        # Consider all items (original logic)
        for i in range(batch_size):
            # Mask out history items
            scores = predicted_scores[i].clone()
            if history_items[i]:
                history_mask = torch.zeros_like(scores, dtype=torch.bool)
                history_mask[history_items[i]] = True
                scores[history_mask] = -float('inf')

            # Extract Top-K
            _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))

            # Find rank of target
            target_positions = (top_k_indices == target_items[i]).nonzero(as_tuple=True)[0]
            if len(target_positions) > 0:
                rank = target_positions[0].item() + 1  # 1-indexed rank
                mrr_scores[i] = 1.0 / rank
    
    return mrr_scores


class RecRewardFrunction:
    """
    TRL-compatible reward function class.
    Integrates with Ray RetrievalService.
    """
    
    def __init__(
        self,
        retrieval_service_name: str = "RetrievalService",
        namespace: str = "rl4rec",
        data_name: str = "beauty",
        reward_type: str = "ndcg",
        k: int = 10,
        normalize: bool = True,
        test_target: bool = False,
    ):
        """
        Args:
            retrieval_service_name: Ray actor name
            namespace: Ray namespace
            data_name: dataset name
            reward_type: reward type ('ndcg', 'hit', 'mrr', 'mixed')
            k: Top-K value
            normalize: whether to normalize rewards
        """
        self.__name__ = "RecRewardFrunction"
        self.retrieval_service_name = retrieval_service_name
        self.namespace = namespace
        self.data_name = data_name
        self.reward_type = reward_type
        self.k = k
        self.normalize = normalize
        self.test_target = test_target

        # Connect to RetrievalService
        try:
            self.retrieval_service = ray.get_actor(
                retrieval_service_name,
                namespace=namespace
            )
            print(f"✓ Connected to {retrieval_service_name}")
        except ValueError as e:
            raise RuntimeError(
                f"Failed to connect to {retrieval_service_name}. "
                f"Make sure retrieval service is running."
            ) from e

        #load item metadata
        with open(f"data/{data_name}/meta_text_fix.json", "r") as f:
            self.item_metadata = json.load(f)
        self.item_metadata = {int(k): v["title"] + "\n" + v["brand"] + "\n" + v["category"] for k, v in self.item_metadata.items()}
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        neg_items: Optional[List[List[int]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        TRL-compatible reward function.
        Uses only the text inside the <query> tag when present.

        Args:
            generated_texts: [batch_size] generated texts
            targets: [batch_size] target item IDs
            histories: [batch_size, *] history item IDs
            neg_items: [batch_size, num_negs] negative item IDs (optional)

        Returns:
            rewards: [batch_size] reward values
        """
        # Extract <query> tag content if present, otherwise use original text
        processed_texts = [extract_query_from_tags(text, tag="query") for text in generated_texts]

        # add target text to generated_texts
        if self.test_target:
            processed_texts = [self.item_metadata[target] + "\n" + processed_text for processed_text, target in zip(processed_texts, targets)]  

        # 1. Compute similarity scores via RetrievalService
        use_negatives_only = neg_items is not None
        
        scores_ref = self.retrieval_service.calculate_reward.remote(
            processed_texts,
            data_name=self.data_name,
            targets=targets if use_negatives_only else None,
            neg_items=neg_items,
        )
        scores = ray.get(scores_ref)  # [batch_size, num_items] or [batch_size, 1+num_negs]
        
        # 2. Compute reward based on reward type
        if self.reward_type == "ndcg":
            rewards = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "hit":
            rewards = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "mrr":
            rewards = calculate_mrr(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        elif self.reward_type == "mixed":
            # Weighted average of NDCG and Hit@K
            ndcg = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
            hit = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
            rewards = 0.7 * ndcg + 0.3 * hit
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        # 3. Normalize (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def compute_metrics(
        self,
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],
        targets: List[int],
        histories: List[List[int]],
        neg_items: Optional[List[List[int]]] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (NDCG, Hit@K, and MRR).

        Args:
            neg_items: [batch_size, num_negs] negative item IDs (optional)

        Returns:
            metrics dictionary
        """
        # Compute similarity scores via RetrievalService
        use_negatives_only = neg_items is not None
        
        scores_ref = self.retrieval_service.calculate_reward.remote(
            completions,
            data_name=self.data_name,
            targets=targets if use_negatives_only else None,
            neg_items=neg_items,
        )
        scores = ray.get(scores_ref)
        
        # Compute all metrics
        ndcg = calculate_ndcg(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        hit = calculate_hit_rate(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        mrr = calculate_mrr(scores, targets, histories, k=self.k, use_negatives_only=use_negatives_only)
        
        return {
            f"ndcg@{self.k}": ndcg.mean().item(),
            f"hit@{self.k}": hit.mean().item(),
            f"mrr@{self.k}": mrr.mean().item(),
        }


# Wrapper function compatible with TRL PPOTrainer
def create_reward_function(
    retrieval_service_name: str = "RetrievalService",
    namespace: str = "rl4rec",
    data_name: str = "beauty",
    reward_type: str = "ndcg",
    k: int = 10,
) -> RecRewardFrunction:
    """
    Create a reward function for use with TRL PPOTrainer
    
    Usage:
        reward_fn = create_reward_function(reward_type="ndcg", k=10)
        rewards = reward_fn(generated_texts, targets, histories)
    """
    return RecRewardFrunction(
        retrieval_service_name=retrieval_service_name,
        namespace=namespace,
        data_name=data_name,
        reward_type=reward_type,
        k=k,
    )


def load_negative_pool(data_name: str, data_dir: str = "data", k: int = 10) -> Dict[int, List[int]]:
    """
    Load negative pool from negative.txt file.

    File format: each line is "user_id neg_item1 neg_item2 ... neg_itemN"

    Args:
        data_name: dataset name (e.g., "beauty")
        data_dir: data directory

    Returns:
        dictionary of negative items lists keyed by user_id
    """
    negative_file = Path(data_dir) / data_name / "negative.txt"
    
    if not negative_file.exists():
        raise FileNotFoundError(f"Negative pool file not found: {negative_file}")
    
    print(f"📦 Loading negative pool from: {negative_file}")
    negative_pool = {}
    
    with open(negative_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user_id = int(parts[0])
            neg_items = [int(item_id) for item_id in parts[1:]]
            #random sample k items
            # neg_items = neg_items[:k-1]
            # neg_items = np.random.choice(neg_items, size=k-1, replace=False).tolist()
            negative_pool[user_id] = neg_items
    
    print(f"✓ Loaded negative pool for {len(negative_pool)} users")
    if len(negative_pool) > 0:
        sample_user = next(iter(negative_pool))
        print(f"  Example: User {sample_user} has {len(negative_pool[sample_user])} negative items")
    
    return negative_pool


class SimilarHistoryItemMentionReward:
    """
    Reward for mentioning the title of the history item most similar to the target.
    Caches the most similar item using embedding similarity; rewards 1 point when the first 3 words of the title are mentioned.
    """
    
    def __init__(
        self,
        data_name: str,
        item_embeddings: torch.Tensor,
        uid_2_target: Dict[int, int],
        device: str = "cuda",
        data_dir: str = "data",
        use_position_weight: bool = False,
        position_decay: float = 1.0,
        similarity_threshold: float = 0.7,
        title_reward_weight: float = 1.0,
        brand_reward_weight: float = 0.5,
        head_range: int = 8,
        # time_window_days: int = -1,
        time_window_days: int = 60,
        decay_ratio: float = 0.7,
    ):
        """
        Args:
            data_name: dataset name
            item_embeddings: pre-computed item embeddings [num_items+1, emb_dim]
            uid_2_target: mapping from user ID to target item ID
            device: device
            data_dir: data directory
            use_position_weight: whether to use position-based weighting
                                when True, earlier mentions receive higher reward
            position_decay: position-based decay rate (0.0 ~ 1.0)
                          0.0 = same reward regardless of position
                          1.0 = reward of 0 at the end of text
                          e.g. 0.5 means reward halves at the end of text
            similarity_threshold: similarity threshold (0.0 ~ 1.0)
                                when below this value, the last interacted item is selected
            time_window_days: use only purchase records within the last N days as anchor candidates
            decay_ratio: decay rate applied per word position after head_range (e.g. 0.95)
        """
        self.__name__ = "SimilarHistoryItemMentionReward"
        self.data_name = data_name
        self.item_embeddings = item_embeddings
        self.device = device
        self.use_position_weight = use_position_weight
        self.position_decay = position_decay
        self.similarity_threshold = similarity_threshold
        self.time_window_days = time_window_days
        
        self.title_reward_weight = title_reward_weight
        self.brand_reward_weight = brand_reward_weight
        self.head_range = head_range
        self.decay_ratio = decay_ratio

        # Load item metadata (title, brand, category)
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            self.item_metadata = json.load(f)
        
        print(f"✓ SimilarHistoryItemMentionReward initialization started")
        print(f"  - Loaded metadata for {len(self.item_metadata)} items")
        
        # Load timestamp information only when time_window_days != -1
        if self.time_window_days != -1:
            print(f"  - Loading user2reviews_with_date.json...")
            with open(f"{data_dir}/{data_name}/user2reviews_with_date.json", "r") as f:
                user_reviews = json.load(f)
            
            # Build user_id -> {item_id: timestamp} mapping
            self.user_item_timestamps = {}
            for user_id_str, reviews in user_reviews.items():
                user_id = int(user_id_str)
                self.user_item_timestamps[user_id] = {}
                for review in reviews:
                    item_id = int(review['item_id'])
                    timestamp = review['timestamp']
                    self.user_item_timestamps[user_id][item_id] = timestamp
            
            print(f"  - Loaded timestamp data for {len(self.user_item_timestamps)} users")
            print(f"  - Time window: {self.time_window_days} days")
        else:
            self.user_item_timestamps = None
            print(f"  - Time window: DISABLED (using all history)")
        
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"    → If max similarity < threshold, use last interacted item")
        if self.use_position_weight:
            print(f"  - Position-based weighting: ENABLED (decay={self.position_decay})")
            print(f"    → Earlier mentions get higher rewards")
        else:
            print(f"  - Position-based weighting: DISABLED")
        
        # Cache: user_id -> (most_similar_history_item_id, max_similarity)
        self.similarity_cache = {}

        # Pre-compute the most similar item for all data
        print(f"  - Pre-computing most similar history items for all users...")
        self._precompute_similar_items(uid_2_target, data_name, data_dir)
        print(f"✓ Pre-computed similar items for {len(self.similarity_cache)} user-target pairs")
    
    def _precompute_similar_items(
        self,
        uid_2_target: Dict[int, int],
        data_name: str,
        data_dir: str
    ):
        """
        Pre-compute the most similar history item to the target for all data.
        When time_window_days != -1, consider only history items within the last N days.
        When similarity is below the threshold, select the last interacted item.

        Args:
            uid_2_target: mapping from user ID to target item ID
            data_name: dataset name
            data_dir: data directory
        """
        # Load history information from sequential_data.txt
        sequential_file = f"{data_dir}/{data_name}/sequential_data.txt"
        
        # Pre-compute normalized embeddings for all items
        normalized_embeddings = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)
        
        fallback_count = 0  # number of times last item was used due to sub-threshold similarity
        time_filtered_count = 0  # number of times history was reduced by time filtering
        no_recent_items_count = 0  # number of times last item used because no items within 60 days
        
        # Disable time filtering when time_window_days == -1
        use_time_filter = (self.time_window_days != -1)
        if use_time_filter:
            time_window_seconds = self.time_window_days * 24 * 60 * 60
        
        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set history
                target_id = parts[-3]   # Train set target

                # Process only users present in uid_2_target
                if user_id not in uid_2_target:
                    continue
                
                # Skip if history is empty
                if len(history) == 0:
                    continue
                
                # Branch based on whether time filtering is enabled
                if use_time_filter:
                    # Get timestamp of the target item
                    if user_id not in self.user_item_timestamps:
                        continue
                    
                    user_timestamps = self.user_item_timestamps[user_id]
                    if target_id not in user_timestamps:
                        continue
                    
                    target_timestamp = int(user_timestamps[target_id])
                    
                    # Keep only history items within the last time_window_days
                    filtered_history = []
                    for item_id in history:
                        if item_id in user_timestamps:
                            item_timestamp = int(user_timestamps[item_id])
                            # Include only items before the target and within the time window
                            if item_timestamp < target_timestamp:
                                time_diff = target_timestamp - item_timestamp
                                if time_diff <= time_window_seconds:
                                    filtered_history.append(item_id)
                    
                    # Use last item from full history if filtered history is empty
                    if len(filtered_history) == 0:
                        selected_item_id = history[-1]
                        max_similarity = 0.0  # fallback due to time filtering
                        self.similarity_cache[user_id] = (selected_item_id, max_similarity)
                        no_recent_items_count += 1
                        continue
                    
                    # Filtering statistics
                    if len(filtered_history) < len(history):
                        time_filtered_count += 1
                    
                    # Use filtered history
                    working_history = filtered_history
                else:
                    # Use full history without time filtering
                    working_history = history
                
                # Target embedding (normalized)
                target_emb = normalized_embeddings[target_id]  # [emb_dim]
                
                # History embeddings (normalized)
                history_ids = torch.tensor(working_history, dtype=torch.long, device=self.device)
                history_embs = normalized_embeddings[history_ids]  # [history_len, emb_dim]
                
                # Compute cosine similarities
                similarities = torch.mm(target_emb.unsqueeze(0), history_embs.T).squeeze(0)  # [history_len]
                
                # Find the most similar item
                max_similarity = similarities.max().item()
                most_similar_idx = similarities.argmax().item()
                
                # Select last interacted item when similarity is below threshold
                if max_similarity < self.similarity_threshold:
                    selected_item_id = working_history[-1]  # last item
                    fallback_count += 1
                else:
                    selected_item_id = working_history[most_similar_idx]
                
                # Save to cache (item ID and max similarity)
                self.similarity_cache[user_id] = (selected_item_id, max_similarity)
        
        # Print statistics
        total_users = len(self.similarity_cache)
        if total_users > 0:
            fallback_ratio = (fallback_count / total_users) * 100
            if use_time_filter:
                time_filtered_ratio = (time_filtered_count / total_users) * 100
                no_recent_ratio = (no_recent_items_count / total_users) * 100
                print(f"  - Time-filtered history: {time_filtered_count}/{total_users} ({time_filtered_ratio:.1f}%)")
                print(f"  - No recent items (use last item): {no_recent_items_count}/{total_users} ({no_recent_ratio:.1f}%)")
                print(f"  - Fallback to last item (within time window): {fallback_count}/{total_users} ({fallback_ratio:.1f}%)")
            else:
                print(f"  - Fallback to last item: {fallback_count}/{total_users} ({fallback_ratio:.1f}%)")

    
    def _get_most_similar_history_item(
        self,
        user_id: int,
    ) -> int:
        """
        Find the history item most similar to the target (from cache or computed on-the-fly).
        Returns the last interacted item when similarity is below the threshold.

        Args:
            user_id: user ID

        Returns:
            selected_item_id: selected history item ID
                            (above threshold: most similar item,
                             below threshold: last interacted item)
        """        
        selected_item_id, _ = self.similarity_cache[user_id]
        return selected_item_id
    
    def _get_first_n_words(self, title: str, n: int) -> str:
        """
        Extract the first n words of a title.

        Args:
            title: item title

        Returns:
            first_n_words: the first n words joined by spaces (lowercase)
        """
        words = title.strip().split()
        first_n_words = " ".join(words[:n])
        return first_n_words.lower()
    
    def _calculate_position_weight(self, position: int, text_length: int) -> float:
        """
        Compute position-based weight.

        Args:
            position: mention position (character index)
            text_length: total text length

        Returns:
            weight: position-based weight (0.0 ~ 1.0)
                   closer to 1.0 at the front, decreases toward the back
        """
        if text_length == 0:
            return 1.0
        
        # Compute relative position (0.0 = start, 1.0 = end)
        position_ratio = position / text_length
        
        # Compute weight: 1.0 - (position_ratio * decay)
        # decay=0.0 → 1.0 regardless of position
        # decay=1.0 → 0.0 at the very end
        # decay=0.5 → 0.5 at the very end
        weight = 1.0 - (position_ratio * self.position_decay)
        
        return max(0.0, weight)  # ensure minimum value of 0.0
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        user_ids: List[int],
        **kwargs
    ) -> List[float]:
        """
        Check whether the title of the most similar history item is mentioned
        in the generated text and return the corresponding reward.
        When a <query> tag is present, only the text inside the tag is inspected.

        Reward decay logic:
        - Mentioned within head_range: apply reward without decay
        - Mentioned after head_range: exponential decay multiplied by decay_ratio per word position
          weight = decay_ratio ^ (word_position - head_range)

        Args:
            generated_texts: [batch_size] generated texts
            targets: [batch_size] target item IDs
            histories: [batch_size, *] history item ID lists
            user_ids: [batch_size] user IDs

        Returns:
            rewards: [batch_size] reward values
                    - within head_range: 0 or full reward (title_reward_weight, brand_reward_weight)
                    - after head_range: exponential decay based on word position
        """
        rewards = []
        
        for gen_text, target_id, history_ids, user_id in zip(generated_texts, targets, histories, user_ids):
            reward = 0.0
            
            # Extract <query> tag content if present, otherwise use original text
            processed_text = extract_query_from_tags(gen_text, tag="query")

            # Find the most similar history item
            most_similar_item_id = self._get_most_similar_history_item(user_id)
            
            # Get the title of the corresponding item
            if str(most_similar_item_id) in self.item_metadata:
                item_title = self.item_metadata[str(most_similar_item_id)]["title"]
                item_brand = self.item_metadata[str(most_similar_item_id)]["brand"]
                first_n_words = self._get_first_n_words(item_title, 3).lower()
                first_n_brand = self._get_first_n_words(item_brand, 3).lower()

                # Check mention position in the full text
                processed_text_lower = processed_text.lower()
                
                # Check title mention
                title_position = processed_text_lower.find(first_n_words)
                if title_position != -1:
                    # Convert character position to word position
                    words_before_title = processed_text_lower[:title_position].split()
                    word_position = len(words_before_title)

                    if word_position < self.head_range:
                        # Within head_range: no decay
                        title_weight = 1.0
                    else:
                        # After head_range: exponential decay
                        words_after_head = word_position - self.head_range
                        title_weight = self.decay_ratio ** words_after_head
                    reward += self.title_reward_weight * title_weight

                # Check brand mention
                brand_position = processed_text_lower.find(first_n_brand)
                if brand_position != -1:
                    # Convert character position to word position
                    words_before_brand = processed_text_lower[:brand_position].split()
                    word_position = len(words_before_brand)

                    if word_position < self.head_range:
                        # Within head_range: no decay
                        brand_weight = 1.0
                    else:
                        # After head_range: exponential decay
                        words_after_head = word_position - self.head_range
                        brand_weight = self.decay_ratio ** words_after_head
                    reward += self.brand_reward_weight * brand_weight
            
            rewards.append(reward)
        
        return rewards


class MetadataMentionReward:
    """
    Reward function that provides higher reward the more the target item's
    metadata (brand, category, etc.) is mentioned.

    Features:
    1. Reward increases the more metadata words are mentioned
    2. Penalty is applied when words from history item metadata absent in the target are mentioned
    3. Reward is normalized inversely proportional to the length of the generated text
    4. Stopwords (none, a, the, etc.) are excluded from reward computation
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
        base_reward: float = 0.1,
        length_penalty_alpha: float = 0.5,
        min_length: int = 10,
        history_penalty_weight: float = 0.01,
    ):
        """
        Args:
            data_name: dataset name
            device: device
            data_dir: data directory
            base_reward: base reward score per metadata word
            length_penalty_alpha: length penalty intensity (0~1; higher values penalize longer text more)
            min_length: minimum text length (no penalty below this length)
            history_penalty_weight: penalty weight for incorrectly mentioning history metadata
        """
        self.__name__ = "MetadataMentionReward"
        self.data_name = data_name
        self.device = device
        self.base_reward = base_reward
        self.length_penalty_alpha = length_penalty_alpha
        self.min_length = min_length
        self.history_penalty_weight = history_penalty_weight
        # Download and load NLTK stopwords (executed only once)
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
        
        # Additional domain-specific stopwords
        additional_stopwords = {'none', 'null', 'n/a', 'na'}
        self.stopwords.update(additional_stopwords)

        # Load item metadata
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        
        # Pre-extract the metadata word set for each item
        self.item_metadata_words = {}
        for item_id, meta in item_metadata.items():
            words = set()
            if "title" in meta and meta["title"]:
                title_words = self._extract_words(str(meta["title"]))
                words.update(title_words)
            
            # Extract brand
            if "brand" in meta and meta["brand"]:
                brand_words = self._extract_words(str(meta["brand"]))
                words.update(brand_words)
            
            # Extract category
            # if "category" in meta and meta["category"]:
            #     category_words = self._extract_words(str(meta["category"]))
            #     words.update(category_words)
            
            self.item_metadata_words[item_id] = words
        
        print(f"✓ MetadataMentionReward initialized")
        print(f"  - Loaded metadata for {len(self.item_metadata_words)} items")
        print(f"  - Base reward: {self.base_reward}")
        print(f"  - Length penalty alpha: {self.length_penalty_alpha}")
        print(f"  - History penalty weight: {self.history_penalty_weight}")
        print(f"  - Stopwords excluded: {len(self.stopwords)} (NLTK English + custom)")
    
    def _extract_words(self, text: str) -> set:
        """
        Extract words from text and remove stopwords.

        Args:
            text: input text

        Returns:
            set of words (lowercase) with stopwords removed
        """
        # Keep only alphanumeric characters and split by whitespace
        import re
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

        # Remove stopwords and single-character words
        filtered_words = {w for w in words if w not in self.stopwords and len(w) > 1}

        return filtered_words

    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        histories: List[List[int]],
        **kwargs
    ) -> List[float]:
        """
        Evaluate how much of the target item's metadata is mentioned in the generated text.
        Penalizes mentions of history item metadata words absent from the target.
        When a <query> tag is present, only the text inside the tag is inspected.

        Args:
            generated_texts: [batch_size] generated texts
            targets: [batch_size] target item IDs
            **kwargs: histories (List[List[int]]): [batch_size, seq_len] history item IDs

        Returns:
            rewards: [batch_size] reward values
        """
        rewards = []
        
        for idx, (gen_text, target_id) in enumerate(zip(generated_texts, targets)):
            # Extract <query> tag content if present, otherwise use original text
            processed_text = extract_query_from_tags(gen_text, tag="query")

            # Target item metadata words
            target_words = self.item_metadata_words.get(target_id, set())
            
            if not target_words:
                rewards.append(0.0)
                continue
            
            # Extract words from processed text
            gen_words = self._extract_words(processed_text)

            # Count how many metadata words are mentioned in the generated text
            matched_words = target_words.intersection(gen_words)
            match_count = len(matched_words)
            
            # Compute base reward (proportional to number of metadata words mentioned)
            reward = match_count * self.base_reward
            
            # Compute history item metadata penalty
            if histories is not None and idx < len(histories):
                history_items = histories[idx]
                
                # Collect metadata words from history items
                history_words = set()
                for hist_id in history_items:
                    hist_words = self.item_metadata_words.get(hist_id, set())
                    history_words.update(hist_words)
                
                # Words that appear only in history, not in target (should not be mentioned)
                wrong_words = history_words - target_words
                
                # Count incorrect words mentioned in the generated text
                wrong_mention_count = len(wrong_words.intersection(gen_words))
                
                # Apply penalty
                penalty = wrong_mention_count * self.history_penalty_weight
                reward = reward - penalty
                reward = max(reward, 0.0)
            
            # Apply length penalty: based on processed text length
            text_length = len(processed_text.split())
            if text_length > self.min_length:
                # length_factor: decreases as text gets longer (0~1)
                length_factor = 1.0 / (1.0 + self.length_penalty_alpha * (text_length - self.min_length) / self.min_length)
                reward = reward * length_factor
            
            rewards.append(reward)
        
        return rewards


class ItemPreferenceMentionReward:
    """
    Reward function that combines target item metadata with inter-item preference
    information to provide rewards.

    Features:
    1. Loads preference information from data_processed/{data_name}_gemma-3-1b-it_item_item_preference.json
    2. Builds a word set by combining existing metadata (brand, category, etc.) with preference text
    3. More shared words in the query yields a higher reward
    4. Stopwords are automatically removed
    """
    
    def __init__(
        self,
        data_name: str,
        device: str = "cuda",
        data_dir: str = "data",
        data_processed_dir: str = "data_processed",
        base_reward: float = 0.1,
        length_penalty_alpha: float = 0.5,
        min_length: int = 10,
    ):
        """
        Args:
            data_name: dataset name
            device: device
            data_dir: metadata directory
            data_processed_dir: item preference data directory
            base_reward: base reward score per word
            length_penalty_alpha: length penalty intensity (0~1; higher values penalize longer text more)
            min_length: minimum text length (no penalty below this length)
        """
        self.__name__ = "ItemPreferenceMentionReward"
        self.data_name = data_name
        self.device = device
        self.base_reward = base_reward
        self.length_penalty_alpha = length_penalty_alpha
        self.min_length = min_length
        
        # Download and load NLTK stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))

        # Additional domain-specific stopwords
        additional_stopwords = {'none', 'null', 'n/a', 'na'}
        self.stopwords.update(additional_stopwords)

        # Load item metadata
        print(f"📦 Loading item metadata from {data_dir}/{data_name}/meta_text_fix.json")
        with open(f"{data_dir}/{data_name}/meta_text_fix.json", "r") as f:
            item_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in item_metadata.items()}
        
        # Load item preference information
        preference_file = f"{data_processed_dir}/{data_name}_gemma-3-1b-it_item_item_preference.json"
        print(f"📦 Loading item preference from {preference_file}")
        with open(preference_file, "r") as f:
            item_preference = json.load(f)
            item_preference = {int(k): v for k, v in item_preference.items()}
        
        # Pre-extract the metadata + preference word set for each item
        self.item_combined_words = {}
        for item_id, meta in item_metadata.items():
            words = set()
            
            # Extract words from metadata
            if "title" in meta and meta["title"]:
                title_words = self._extract_words(str(meta["title"]))
                words.update(title_words)
            
            if "brand" in meta and meta["brand"]:
                brand_words = self._extract_words(str(meta["brand"]))
                words.update(brand_words)
            
            # Extract words from preference information
            if item_id in item_preference:
                preference_words = self._extract_words(item_preference[item_id])
                words.update(preference_words)
            
            self.item_combined_words[item_id] = words
        
        print(f"✓ ItemPreferenceMentionReward initialized")
        print(f"  - Loaded metadata for {len(item_metadata)} items")
        print(f"  - Loaded preference for {len(item_preference)} items")
        print(f"  - Combined word sets created for {len(self.item_combined_words)} items")
        print(f"  - Base reward: {self.base_reward}")
        print(f"  - Length penalty alpha: {self.length_penalty_alpha}")
        print(f"  - Stopwords excluded: {len(self.stopwords)} (NLTK English + custom)")
        
        # Print statistics
        if len(self.item_combined_words) > 0:
            word_counts = [len(words) for words in self.item_combined_words.values()]
            avg_words = sum(word_counts) / len(word_counts)
            max_words = max(word_counts)
            min_words = min(word_counts)
            print(f"  - Word set statistics: Min={min_words}, Max={max_words}, Avg={avg_words:.1f}")
    
    def _extract_words(self, text: str) -> set:
        """
        Extract words from text and remove stopwords.

        Args:
            text: input text

        Returns:
            Set of words with stopwords removed (lowercase)
        """
        # Keep only alphanumeric characters separated by spaces
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

        # Remove stopwords and single-character words
        filtered_words = {w for w in words if w not in self.stopwords and len(w) > 1}
        
        return filtered_words
    
    def __call__(
        self,
        generated_texts: List[str],
        targets: List[int],
        **kwargs
    ) -> List[float]:
        """
        Evaluate how many target item metadata+preference words are mentioned
        in generated text and return rewards.
        If a <query> tag is present, only the text inside the tag is inspected.

        Args:
            generated_texts: [batch_size] generated texts
            targets: [batch_size] target item IDs

        Returns:
            rewards: [batch_size] reward values
        """
        rewards = []

        for gen_text, target_id in zip(generated_texts, targets):
            # Target item metadata + preference words
            target_words = self.item_combined_words.get(target_id, set())

            if not target_words:
                rewards.append(0.0)
                continue

            # Extract words from processed text
            gen_words = self._extract_words(gen_text)

            # Count how many shared words are mentioned
            matched_words = target_words.intersection(gen_words)
            match_count = len(matched_words)

            # Compute base reward (proportional to number of words mentioned)
            reward = match_count * self.base_reward

            # Apply length penalty: based on generated text length
            text_length = len(gen_text.split())
            if text_length > self.min_length:
                # length_factor: decreases as text gets longer (0~1)
                length_factor = 1.0 / (1.0 + self.length_penalty_alpha * (text_length - self.min_length) / self.min_length)
                reward = reward * length_factor
            
            rewards.append(reward)
        
        return rewards


class FormatComplianceReward:
    """
    Reward function that checks whether generated text follows a specific XML-like format.

    Required format:
    <thinking>...</thinking>
    <window>...</window>
    <items>...</items>
    <query>...</query>

    Partial credit is given when each tag exists and is correctly opened and closed.
    """
    
    def __init__(
        self,
        required_tags: List[str] = None,
        reward_per_tag: float = 0.25,
        strict_order: bool = False,
        case_sensitive: bool = False,
    ):
        """
        Args:
            required_tags: list of required tags (default: ["thinking", "window", "items", "query"])
            reward_per_tag: reward score per tag (default: 0.25; 4 tags * 0.25 = 1.0)
            strict_order: whether to strictly check tag ordering (default: False)
            case_sensitive: whether to differentiate case (default: False)
        """
        self.__name__ = "FormatComplianceReward"
        
        if required_tags is None:
            self.required_tags = ["thinking", "window", "items", "query"]
        else:
            self.required_tags = required_tags
        
        self.reward_per_tag = reward_per_tag
        self.strict_order = strict_order
        self.case_sensitive = case_sensitive
        
        print(f"✓ FormatComplianceReward initialized")
        print(f"  - Required tags: {self.required_tags}")
        print(f"  - Reward per tag: {self.reward_per_tag}")
        print(f"  - Strict order: {self.strict_order}")
        print(f"  - Case sensitive: {self.case_sensitive}")
        print(f"  - Max reward: {len(self.required_tags) * self.reward_per_tag}")
    
    def _check_tag_exists(self, text: str, tag: str) -> bool:
        """
        Check whether a tag is correctly opened and closed.

        Args:
            text: text to inspect
            tag: tag name (e.g. "thinking")
            
        Returns:
            True if both opening and closing tags exist, False otherwise
        """
        if not self.case_sensitive:
            text = text.lower()
            tag = tag.lower()
        
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        
        return open_tag in text and close_tag in text
    
    def _check_tag_order(self, text: str) -> bool:
        """
        Check whether tags appear in the correct order.

        Args:
            text: text to inspect
            
        Returns:
            True if tags appear in correct order, False otherwise
        """
        if not self.case_sensitive:
            text = text.lower()
        
        last_position = -1
        
        for tag in self.required_tags:
            open_tag = f"<{tag}>"
            if not self.case_sensitive:
                tag = tag.lower()
            
            position = text.find(open_tag)
            
            if position == -1:
                return False
            
            if position < last_position:
                return False
            
            last_position = position
        
        return True
    
    def __call__(
        self,
        generated_texts: List[str],
        **kwargs
    ) -> List[float]:
        """
        Evaluate format compliance of generated texts and return rewards.

        Args:
            generated_texts: [batch_size] generated texts

        Returns:
            rewards: [batch_size] reward values (0.0 ~ max_reward)
        """
        rewards = []

        for gen_text in generated_texts:
            reward = 0.0

            # 1. Check whether each tag exists
            valid_tags = 0
            for tag in self.required_tags:
                if self._check_tag_exists(gen_text, tag):
                    valid_tags += 1
                    reward += self.reward_per_tag

            # 2. Strict order check (optional)
            if self.strict_order and valid_tags > 0:
                if not self._check_tag_order(gen_text):
                    # Halve the reward if tag order is incorrect
                    reward = reward * 0.5
            
            rewards.append(reward)
        
        return rewards


class LocalEmbeddingRewardFunction:
    """
    Local embedding-based reward function.
    Computes negative item embeddings internally (without a DB) and uses NDCG as the reward.
    """
    
    def __init__(
        self,
        args: argparse.Namespace,
        uid_2_target: Dict[int, int],
    ):
        """
        Args:
            uid_2_target: user ID to target item ID mapping
            data_name: dataset name
            k: Top-K value
            reward_type: reward type ('ndcg', 'hit', 'mrr', 'mixed')
            emb_model_name: embedding model name
            emb_type: embedding type (review_description, title, etc.)
            device: device
            emb_batch_size: batch size for embedding computation
            data_dir: data directory
            normalize: whether to normalize rewards
            novelty_reward: whether to enable Novelty reward (True/False)
                           Novelty = NDCG × popularity_weight
            novelty_coef: Novelty reward coefficient (default: 1.0)
            novelty_target_rank: (unused, backward compatibility)
            novelty_mode: (unused, backward compatibility)
            novelty_annealing: whether to use Novelty annealing
                              If True, the novelty ratio increases linearly from 0→1 as training progresses
                              Final reward = (1-ratio)*base + ratio*novelty_coef*novelty
            popularity_coef: Popularity reward coefficient (0.0 = disabled)
                            Popularity bonus is added only when the prediction is correct
            target_emb_reward: whether to use target embedding similarity reward
            target_emb_coef: target embedding reward coefficient
            infonce_reward: whether to use InfoNCE (contrastive learning) reward
            infonce_coef: InfoNCE reward coefficient
            infonce_temperature: InfoNCE temperature parameter (default: 0.07)
            infonce_emb_type: embedding type for InfoNCE (if None, same as emb_type)
            proxy_label_reward: whether to use Proxy label reward
                               If True, top proxy_k items similar to the target are also treated as partial correct labels
                               Added on top of the existing base_reward
            proxy_k: number of similar items to use as proxy labels
            proxy_label_coef: Proxy label reward coefficient
            proxy_label_cutoff: Proxy label similarity threshold (default: 0.0)
                               Items with similarity below this value are excluded from proxy labels
                               e.g., setting 0.5 filters out items with similarity < 0.5
            max_steps: maximum training steps (used for novelty annealing calculation)
        """
        self.__name__ = "LocalEmbeddingRewardFunction"
        self.args = args
        self.data_name = args.data_name
        self.reward_type = args.reward_type
        self.k = args.k
        self.normalize = args.normalize_rewards
        self.device = args.device
        self.emb_batch_size = args.emb_batch_size
        self.uid_2_target = uid_2_target  # Store for full item pool ranking
        
        # Novelty-related parameters
        self.novelty_reward = args.novelty_reward
        self.novelty_coef = args.novelty_coef
        self.novelty_target_rank = args.novelty_target_rank
        self.novelty_mode = args.novelty_mode
        self.novelty_annealing = args.novelty_annealing
        
        # Popularity-related parameters
        self.popularity_coef = args.popularity_coef
        
        # Target embedding similarity reward parameters
        self.target_emb_reward = args.target_emb_reward
        self.target_emb_file = args.target_emb_file
        self.target_emb_coef = args.target_emb_coef
        
        # InfoNCE reward parameters
        self.infonce_reward = args.infonce_reward
        self.infonce_coef = args.infonce_coef
        self.infonce_temperature = args.infonce_temperature
        self.infonce_emb_type = args.infonce_emb_type if args.infonce_emb_type is not None else args.emb_type
        
        # Proxy label reward parameters
        if hasattr(args, "proxy_label_reward"):
            self.proxy_label_reward = args.proxy_label_reward
            self.proxy_k = args.proxy_k
            self.proxy_label_coef = args.proxy_label_coef
            self.proxy_label_cutoff = args.proxy_label_cutoff if hasattr(args, "proxy_label_cutoff") else 0.98
        else:
            self.proxy_label_reward = False
            self.proxy_k = 0
            self.proxy_label_coef = 0
            self.proxy_label_cutoff = 0.0
        
        # Anchor-Guided GRPO parameters
        if hasattr(args, "anchor_reward"):
            self.anchor_reward = args.anchor_reward
            self.anchor_coef = args.anchor_coef
            self.anchor_radius_start = args.anchor_radius_start
            self.anchor_radius_end = args.anchor_radius_end
            self.anchor_penalty_mode = args.anchor_penalty_mode
            self.anchor_penalty_value = args.anchor_penalty_value
        else:
            self.anchor_reward = False
            self.anchor_coef = 1.0
            self.anchor_radius_start = 0.5
            self.anchor_radius_end = 1.0
            self.anchor_penalty_mode = "soft"
            self.anchor_penalty_value = -1.0
        
        # Adaptive Threshold Reward parameters
        if hasattr(args, "adaptive_threshold_reward"):
            self.adaptive_threshold_reward = args.adaptive_threshold_reward
            self.adaptive_threshold_coef = args.adaptive_threshold_coef
            self.adaptive_tau_min = args.adaptive_tau_min
        else:
            self.adaptive_threshold_reward = False
            self.adaptive_threshold_coef = 1.0
            self.adaptive_tau_min = 0.0
        
        # History Proxy Threshold Reward parameters
        if hasattr(args, "history_proxy_threshold_reward"):
            self.history_proxy_threshold_reward = args.history_proxy_threshold_reward
            self.history_proxy_threshold_coef = args.history_proxy_threshold_coef
        else:
            self.history_proxy_threshold_reward = False
            self.history_proxy_threshold_coef = 1.0
        
        # Training-related parameters
        self.max_steps = args.max_steps

        # Variables for tracking reward decomposition (for wandb logging)
        self.last_base_rewards = None
        self.last_proxy_label_rewards = None
        self.last_target_emb_rewards = None
        self.last_infonce_rewards = None
        self.last_anchor_rewards = None
        self.last_adaptive_threshold_rewards = None
        self.last_history_proxy_threshold_rewards = None
        
        print(f"💰 Reward configuration:")
        print(f"  - Reward type: {self.reward_type}")
        print(f"  - Top-K: {self.k}")
        print(f"  - Normalize: {self.normalize}")
        if self.novelty_reward:
            print(f"  - Novelty reward: ENABLED")
            print(f"  - Novelty coefficient: {self.novelty_coef}")
            print(f"  - Novelty = NDCG × popularity_weight (encourages unpopular items)")
            if self.novelty_annealing:
                print(f"  - Novelty annealing: ENABLED")
                print(f"  - Novelty ratio will increase linearly from 0 to 1 over {self.max_steps} steps")
                print(f"  - Final reward = (1-ratio)*base + ratio*novelty")
        if self.popularity_coef > 0:
            print(f"  - Popularity coefficient: {self.popularity_coef}")
            print(f"  - Popularity bonus for unpopular items (when correct)")
        if self.target_emb_reward:
            print(f"  - Target embedding reward: ENABLED")
            print(f"  - Target embedding file: {self.target_emb_file}")
            print(f"  - Target embedding coefficient: {self.target_emb_coef}")
            print(f"  - Reward based on cosine similarity with target embedding")
        if self.infonce_reward:
            print(f"  - InfoNCE reward: ENABLED")
            print(f"  - InfoNCE coefficient: {self.infonce_coef}")
            print(f"  - InfoNCE temperature: {self.infonce_temperature}")
            print(f"  - InfoNCE embedding type: {self.infonce_emb_type}")
            print(f"  - Contrastive learning: maximize target similarity, minimize negative similarity")
        if self.proxy_label_reward:
            print(f"  - Proxy label reward: ENABLED")
            print(f"  - Proxy K: {self.proxy_k}")
            print(f"  - Proxy label coefficient: {self.proxy_label_coef}")
            print(f"  - Proxy label cutoff: {self.proxy_label_cutoff}")
            print(f"    → Items with similarity < {self.proxy_label_cutoff} will be excluded from proxy labels")
            print(f"  - Use top-{self.proxy_k} similar items as soft labels with similarity-weighted NDCG")
            print(f"  - Final reward = base_reward + proxy_label_coef * proxy_label_ndcg")
        if self.anchor_reward:
            print(f"  - Anchor-Guided GRPO (AG-GRPO): ENABLED")
            print(f"  - Anchor coefficient: {self.anchor_coef}")
            print(f"  - Anchor radius: {self.anchor_radius_start} → {self.anchor_radius_end} (curriculum learning)")
            print(f"  - Penalty mode: {self.anchor_penalty_mode}")
            if self.anchor_penalty_mode == "hard":
                print(f"  - Hard penalty value: {self.anchor_penalty_value}")
            print(f"  - Reward based on similarity with last item (anchor) embedding")
            print(f"  - Gradually expands exploration radius as training progresses")
        if self.adaptive_threshold_reward:
            print(f"  - Adaptive Threshold Reward: ENABLED")
            print(f"  - Adaptive threshold coefficient: {self.adaptive_threshold_coef}")
            print(f"  - Minimum threshold (tau_min): {self.adaptive_tau_min}")
            print(f"  - Uses dynamic threshold based on historical item similarity (S_base)")
            print(f"  - Reward = 1 if CosSim(query, target) > max(tau_min, S_base), else 0")
        if self.history_proxy_threshold_reward:
            print(f"  - History Proxy Threshold Reward: ENABLED")
            print(f"  - History proxy threshold coefficient: {self.history_proxy_threshold_coef}")
            print(f"  - Uses most similar history item to target as proxy")
            print(f"  - Reward = max(0, CosSim(query, proxy) - mean(CosSim(query, other_history)))")
        
        # Load embedding model
        print(f"🤖 Loading embedding model: {args.emb_model_name}")
        from sentence_transformers import SentenceTransformer
        self.emb_model = SentenceTransformer(args.emb_model_name, device=self.device)
        print(f"✓ Embedding model loaded on {self.device}")

        total_user_count = 0
        sequential_file = f"data/{self.data_name}/sequential_data.txt"        
        with open(sequential_file, 'r') as f:
            for line in f:
                total_user_count += 1
        
        # Use full item pool if k > 100, otherwise use negative pool
        self.use_full_item_pool = (self.k > 100)
        if self.use_full_item_pool:
            print(f"⚠️ k={self.k} > 100: Using full item pool for ranking (no negative sampling)")
            self.negative_pool = None
            self.candidate_tensor = None  # Will use full item embeddings
        else:
            # Load negative pool
            self.negative_pool = load_negative_pool(self.data_name, args.data_dir, self.k)
            # prepare candidate set, target comes first
            self.candidate_tensor = self._prepare_candidate_tensor(total_user_count, uid_2_target, self.negative_pool)
        
        # Load pre-computed item embeddings
        emb_model_name_dir = args.emb_model_name.split("/")[-1]
        item_embedding_file_path = f"data_emb/{self.data_name}_{args.emb_type}_{emb_model_name_dir}_emb.pt"
        print(f"📦 Loading pre-computed item embeddings from: {item_embedding_file_path}")
        self.item_embeddings = torch.load(item_embedding_file_path, map_location=self.device)
        print(f"✓ Loaded embeddings for {len(self.item_embeddings)} items")
        
        # Load additional embeddings for InfoNCE (if needed)
        if self.infonce_reward and self.infonce_emb_type != args.emb_type:
            infonce_embedding_file_path = f"data_emb/{self.data_name}_{self.infonce_emb_type}_{emb_model_name_dir}_emb.pt"
            print(f"📦 Loading InfoNCE embeddings from: {infonce_embedding_file_path}")
            self.infonce_item_embeddings = torch.load(infonce_embedding_file_path, map_location=self.device)
            print(f"✓ Loaded InfoNCE embeddings for {len(self.infonce_item_embeddings)} items")
        else:
            # Use the same embeddings
            self.infonce_item_embeddings = self.item_embeddings if self.infonce_reward else None
        
        # Load or compute inter-item similarity for proxy labels
        if self.proxy_label_reward:
            # Check for saved proxy labels file
            # Use args.proxy_label_file if specified, otherwise auto-generate
            if hasattr(args, 'proxy_label_file') and args.proxy_label_file is not None:
                proxy_labels_file = args.proxy_label_file
                print(f"📦 Using user-specified proxy labels file: {proxy_labels_file}")
            else:
                proxy_labels_file = f"data_emb/{self.data_name}_proxy_labels_k100_{args.emb_type}_{emb_model_name_dir}.json"
                print(f"📦 Using auto-generated proxy labels path: {proxy_labels_file}")
            
            proxy_labels_path = Path(proxy_labels_file)
            
            if proxy_labels_path.exists():
                print(f"✓ Loading pre-computed proxy labels from: {proxy_labels_file}")
                self.item_proxy_labels = self._load_proxy_labels(proxy_labels_path)
                print(f"✓ Loaded proxy labels for {len(self.item_proxy_labels)} items")
            else:
                print(f"⚠️  Pre-computed proxy labels not found: {proxy_labels_file}")
                print(f"   Computing proxy labels on-the-fly (this may take time)...")
                exit()

        else:
            self.item_proxy_labels = None
        
        # Prepare target embeddings (when target_emb_reward is used)
        if self.target_emb_reward:
            self.target_embeddings = self._prepare_target_embeddings(uid_2_target)
            print(f"✓ Prepared target embeddings for {len(uid_2_target)} users")
        else:
            self.target_embeddings = None
        
        # Prepare last item (anchor) embeddings (when anchor_reward is used)
        if self.anchor_reward:
            self.last_item_embeddings = self._prepare_last_item_embeddings()
            print(f"✓ Prepared last item (anchor) embeddings for anchor-guided exploration")
        else:
            self.last_item_embeddings = None
        
        # Prepare history items (when adaptive_threshold_reward is used)
        if self.adaptive_threshold_reward:
            self.user_history_items = self._prepare_user_history_items()
            print(f"✓ Prepared user history items for adaptive threshold reward")
        else:
            self.user_history_items = None
        
        # Prepare history proxy items (when history_proxy_threshold_reward is used)
        if self.history_proxy_threshold_reward:
            self.user_history_items = self._prepare_user_history_items()
            self.user_history_proxy_items = self._prepare_user_history_proxy_items(uid_2_target)
            print(f"✓ Prepared user history proxy items for history proxy threshold reward")
        else:
            self.user_history_proxy_items = None
        
        # Compute item popularity from train set
        # Required when novelty or popularity reward is enabled
        if self.novelty_reward or self.popularity_coef > 0:
            self.item_popularity_weights = self._compute_item_popularity(
                uid_2_target, self.negative_pool, self.data_name, args.data_dir
            )
        else:
            self.item_popularity_weights = None

    def _prepare_candidate_tensor(self, total_user_count: int, uid_2_target: Dict[int, int], neg_pool: Dict[int, List[int]]) -> torch.Tensor:
        candidate_tensor = torch.zeros(total_user_count+1, len(list(neg_pool.values())[0])+1, dtype=torch.long)
        for uid, target_id in uid_2_target.items():
            candidate_tensor[uid] = torch.tensor([target_id] + neg_pool[uid], dtype=torch.long)
        return candidate_tensor
    
    def _load_proxy_labels(self, file_path: Path) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Load saved proxy labels.

        Args:
            file_path: path to the proxy labels JSON file

        Returns:
            item_proxy_labels: Dict[item_id, (proxy_item_ids, proxy_similarities)]
        """
        with open(file_path, 'r') as f:
            proxy_labels_json = json.load(f)

        # Convert data loaded from JSON into tensors
        item_proxy_labels = {}
        total_filtered = 0  # statistics for filtered proxy count

        for item_id_str, proxy_list in proxy_labels_json.items():
            item_id = int(item_id_str)

            # 1. Truncate to proxy_k entries
            proxy_list = proxy_list[:self.proxy_k]

            # 2. Filter items below the cutoff threshold
            if self.proxy_label_cutoff > 0:
                filtered_proxy_list = [(pid, sim) for pid, sim in proxy_list if sim >= self.proxy_label_cutoff]
                total_filtered += len(proxy_list) - len(filtered_proxy_list)
                proxy_list = filtered_proxy_list

            # 3. Save only if proxies remain after filtering
            if len(proxy_list) > 0:
                # Split List[Tuple[item_id, similarity]] into two tensors
                proxy_ids = torch.tensor([p[0] for p in proxy_list], dtype=torch.long, device=self.device)
                proxy_sims = torch.tensor([p[1] for p in proxy_list], dtype=torch.float32, device=self.device)

                item_proxy_labels[item_id] = (proxy_ids, proxy_sims)

        # Print filtering statistics
        if self.proxy_label_cutoff > 0:
            print(f"  - Filtered {total_filtered} proxy labels below cutoff {self.proxy_label_cutoff}")
            avg_proxies = sum(len(v[0]) for v in item_proxy_labels.values()) / max(1, len(item_proxy_labels))
            print(f"  - Average proxies per item after filtering: {avg_proxies:.2f}")
        
        return item_proxy_labels
    
    def _prepare_target_embeddings(self, uid_2_target: Dict[int, int]) -> torch.Tensor:
        """
        Prepare target item embeddings for each user.

        Args:
            uid_2_target: user ID to target item ID mapping

        Returns:
            target_embeddings: [max_uid+1, emb_dim] target embedding for each user
        """
        if self.target_emb_file is not None:
            target_embeddings = torch.load(f"data_emb/{self.args.target_emb_file}", map_location=self.device)
            return target_embeddings

        max_uid = max(uid_2_target.keys())
        emb_dim = self.item_embeddings.shape[1]

        # Initialize per-user target embedding tensor
        target_embeddings = torch.zeros(max_uid + 1, emb_dim, device=self.device)
        
        for uid, target_id in uid_2_target.items():
            target_embeddings[uid] = self.item_embeddings[target_id]
        
        return target_embeddings
    
    def _prepare_last_item_embeddings(self) -> torch.Tensor:
        """
        Prepare the last item (anchor) embedding for each user.
        Reads the last item based on the validation set from sequential_data.txt.

        Returns:
            last_item_embeddings: [max_uid+1, emb_dim] last item embedding for each user
        """
        print(f"📦 Preparing last item (anchor) embeddings from sequential data...")

        sequential_file = f"data/{self.data_name}/sequential_data.txt"

        # Collect last item ID per user
        uid_2_last_item = {}
        max_uid = 0

        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                last_item_id = parts[-3]  # last item
                uid_2_last_item[user_id] = last_item_id
                max_uid = max(max_uid, user_id)
        
        # Initialize embedding tensor
        emb_dim = self.item_embeddings.shape[1]
        last_item_embeddings = torch.zeros(max_uid + 1, emb_dim, device=self.device)

        # Fill in last item embeddings
        for uid, last_item_id in uid_2_last_item.items():
            last_item_embeddings[uid] = self.item_embeddings[last_item_id]
        
        print(f"  Total users with last item: {len(uid_2_last_item)}")
        print(f"  Max user ID: {max_uid}")
        
        return last_item_embeddings
    
    def _prepare_user_history_items(self) -> Dict[int, torch.Tensor]:
        """
        Prepare the list of past purchased items for each user (for adaptive threshold reward).
        Reads train set history from sequential_data.txt.

        Returns:
            user_history_items: Dict[user_id, history_item_ids_tensor]
        """
        print(f"📦 Preparing user history items from sequential data...")

        sequential_file = f"data/{self.data_name}/sequential_data.txt"

        # Collect history items per user
        user_history_items = {}

        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set history

                # Skip if history is empty
                if len(history) == 0:
                    continue

                # Convert to tensor and store
                user_history_items[user_id] = torch.tensor(history, dtype=torch.long, device=self.device)

        print(f"  Total users with history: {len(user_history_items)}")

        # Print statistics
        if len(user_history_items) > 0:
            history_lengths = [len(h) for h in user_history_items.values()]
            avg_length = sum(history_lengths) / len(history_lengths)
            min_length = min(history_lengths)
            max_length = max(history_lengths)
            print(f"  History length - Min: {min_length}, Max: {max_length}, Avg: {avg_length:.2f}")
        
        return user_history_items
    
    def _prepare_user_history_proxy_items(self, uid_2_target: Dict[int, int]) -> Dict[int, int]:
        """
        Pre-compute the history item most similar to the target item for each user.
        (For history proxy threshold reward.)

        Strategy:
        - Compute cosine similarity between the target item and each history item
        - Store the history item with the highest similarity as the proxy

        Args:
            uid_2_target: user ID to target item ID mapping

        Returns:
            user_history_proxy_items: Dict[user_id, proxy_item_id]
        """
        print(f"📦 Pre-computing most similar history items to target for each user...")

        user_history_proxy_items = {}

        # Normalize item embeddings (for cosine similarity computation)
        item_embeddings_norm = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)

        users_with_proxy = 0
        users_without_history = 0

        for uid, target_id in uid_2_target.items():
            # Skip if user has no history
            if uid not in self.user_history_items:
                users_without_history += 1
                continue

            history_item_ids = self.user_history_items[uid]  # [history_len]

            # Target embedding
            target_emb = item_embeddings_norm[target_id].unsqueeze(0)  # [1, emb_dim]

            # History embeddings
            history_embs = item_embeddings_norm[history_item_ids]  # [history_len, emb_dim]

            # Compute similarity between target and history items
            similarities = torch.mm(target_emb, history_embs.T).squeeze(0)  # [history_len]

            # Select history item with highest similarity
            max_sim_idx = similarities.argmax().item()
            proxy_item_id = history_item_ids[max_sim_idx].item()
            max_similarity = similarities[max_sim_idx].item()

            user_history_proxy_items[uid] = proxy_item_id
            users_with_proxy += 1

            # Debug: print info for the first 5 users
            if users_with_proxy <= 5:
                print(f"  User {uid}: Target={target_id}, Proxy={proxy_item_id}, "
                      f"Similarity={max_similarity:.4f}, History len={len(history_item_ids)}")
        
        print(f"  Total users with proxy: {users_with_proxy}")
        print(f"  Users without history: {users_without_history}")
        
        return user_history_proxy_items
    
    def _compute_item_popularity(
        self, 
        uid_2_target: Dict[int, int], 
        neg_pool: Dict[int, List[int]],
        data_name: str,
        data_dir: str
    ) -> torch.Tensor:
        """
        Compute item popularity from the train set (using inverse log).

        Args:
            uid_2_target: user ID to target item ID mapping
            neg_pool: Negative pool
            data_name: dataset name
            data_dir: data directory

        Returns:
            item_popularity_weights: [num_items] popularity weight per item;
                                     higher value for less popular items
        """
        print(f"📊 Computing item popularity weights from train set...")

        # Load train set from sequential_data.txt (target_index=-3)
        sequential_file = f"{data_dir}/{data_name}/sequential_data.txt"
        item_counts = {}

        with open(sequential_file, 'r') as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:-3]  # Train set history
                target = parts[-3]  # Train set target

                # Count all items in history
                for item_id in history:
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1

                # Count target item as well
                item_counts[target] = item_counts.get(target, 0) + 1

        # Determine total number of items
        all_item_ids = set(item_counts.keys())
        max_item_id = max(all_item_ids) if all_item_ids else 0
        
        print(f"  Total unique items in train set: {len(all_item_ids)}")
        print(f"  Max item ID: {max_item_id}")
        
        # Compute popularity weights: inverse of log(count + 1)
        # Popular items -> lower weight
        # Unpopular items -> higher weight
        item_weights = torch.ones(max_item_id + 1, device=self.device)

        for item_id, count in item_counts.items():
            # Inverse of log(count + 1)
            item_weights[item_id] = 1.0 / np.log(count + 1)

        # Normalize so that the mean equals 1
        # Items that never appear keep weight 1.0
        appeared_mask = torch.zeros(max_item_id + 1, dtype=torch.bool, device=self.device)
        for item_id in all_item_ids:
            appeared_mask[item_id] = True
        
        if appeared_mask.sum() > 0:
            mean_weight = item_weights[appeared_mask].mean()
            item_weights[appeared_mask] = item_weights[appeared_mask] / mean_weight
        
        print(f"  Item popularity weight statistics:")
        print(f"    Min: {item_weights[appeared_mask].min().item():.4f}")
        print(f"    Max: {item_weights[appeared_mask].max().item():.4f}")
        print(f"    Mean: {item_weights[appeared_mask].mean().item():.4f}")
        print(f"    Std: {item_weights[appeared_mask].std().item():.4f}")
        
        # Print examples
        sorted_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-5 popular items:")
        for item_id, count in sorted_counts[:5]:
            weight = item_weights[item_id].item()
            print(f"    Item {item_id}: count={count}, weight={weight:.4f}")
        
        print(f"  Bottom-5 popular items:")
        for item_id, count in sorted_counts[-5:]:
            weight = item_weights[item_id].item()
            print(f"    Item {item_id}: count={count}, weight={weight:.4f}")
        
        print(f"✓ Item popularity weights computed")
        
        return item_weights
    
    def _encode_texts(self, generated_texts: List[str]) -> torch.Tensor:
        """
        Convert generated texts to embeddings.
        If a <query> tag is present, only the text inside the tag is used.

        Args:
            generated_texts: [batch_size] generated texts

        Returns:
            embeddings: [batch_size, emb_dim] embeddings
        """
        # Extract <query> tag content if present, otherwise use original text
        processed_texts = [extract_query_from_tags(text, tag="query") for text in generated_texts]
        
        embeddings = self.emb_model.encode(
            processed_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
            batch_size=self.emb_batch_size,
        )
        return embeddings
    
    def _compute_similarity_scores(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute similarity scores between query embeddings and candidate set
        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user ids
            return_scores: if True, return (ranks, scores), otherwise return (ranks, None)
        Returns:
            ranks: [batch_size] ranks of target items
            scores: [batch_size, num_candidates] similarity scores (only if return_scores=True)
        """
        if self.use_full_item_pool:
            # Compute similarity over the full item pool
            # query_embeddings: [batch_size, emb_dim]
            # item_embeddings: [num_items, emb_dim]
            scores = torch.mm(query_embeddings, self.item_embeddings.T)  # [batch_size, num_items]

            # Retrieve target item for each user
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]

            # Score of the target item
            target_scores = scores[torch.arange(scores.size(0), device=self.device), target_item_ids]  # [batch_size]

            # Rank: number of items with a higher score than target + 1
            ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
        else:
            # Negative pool-based computation (original logic)
            batch_candidate_tensor = self.candidate_tensor[user_ids]
            scores = torch.bmm(query_embeddings.unsqueeze(1), self.item_embeddings[batch_candidate_tensor].transpose(1, 2)).squeeze(1)
            target_scores = scores[:, 0].unsqueeze(1)
            ranks = (scores > target_scores).sum(dim=1) + 1
        
        if return_scores:
            return ranks, scores
        else:
            return ranks, None
    
    def _compute_proxy_label_ndcg(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        predicted_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NDCG based on proxy labels.
        Uses pre-computed inter-item similarity to treat target proxy labels as soft labels.

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs
            predicted_scores: [batch_size, num_items] or [batch_size, num_candidates] predicted scores

        Returns:
            ndcg_rewards: [batch_size] NDCG reward based on proxy labels
        """
        batch_size = len(user_ids)
        ndcg_rewards = torch.zeros(batch_size, device=self.device)

        # Retrieve target item IDs
        target_item_ids = torch.tensor(
            [self.uid_2_target[uid] for uid in user_ids],
            device=self.device
        )  # [batch_size]
        
        if self.use_full_item_pool:
            # Use full item pool
            num_items = len(self.item_embeddings)

            for i in range(batch_size):
                target_id = target_item_ids[i].item()

                proxy_ids, proxy_sims = self.item_proxy_labels[target_id]

                # Build relevance scores: target itself = 1.0, proxies proportional to similarity
                relevance_scores = torch.zeros(num_items, device=self.device)
                relevance_scores[target_id] = 1.0  # target itself
                relevance_scores[proxy_ids] = proxy_sims  # Proxy labels

                # Extract Top-K based on predicted scores
                pred_scores = predicted_scores[i]  # [num_items]
                top_k_pred_scores, top_k_pred_indices = torch.topk(pred_scores, k=min(self.k, len(pred_scores)))

                # Extract relevance from Top-K predictions
                predicted_relevance = relevance_scores[top_k_pred_indices]  # [k]

                # Compute DCG
                dcg = calculate_dcg(predicted_relevance.unsqueeze(0), k=self.k)[0]

                # Compute IDCG (ideal case: sorted by relevance descending)
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_relevance = ideal_relevance[:self.k]
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=self.k)[0]

                # Compute NDCG
                if idcg > 0:
                    ndcg_rewards[i] = dcg / (idcg + 1e-10)
                else:
                    ndcg_rewards[i] = 0.0
        else:
            # Candidate set-based computation
            for i in range(batch_size):
                target_id = target_item_ids[i].item()

                # Retrieve candidate set
                batch_candidate_tensor = self.candidate_tensor[user_ids[i]]  # [num_candidates]
                num_candidates = len(batch_candidate_tensor)

                # Retrieve pre-computed proxy labels for target item
                if target_id not in self.item_proxy_labels:
                    # If no proxy labels, only target gets 1.0
                    proxy_ids = torch.tensor([], dtype=torch.long, device=self.device)
                    proxy_sims = torch.tensor([], dtype=torch.float32, device=self.device)
                else:
                    proxy_ids, proxy_sims = self.item_proxy_labels[target_id]

                # Build relevance scores within the candidate set
                relevance_scores = torch.zeros(num_candidates, device=self.device)

                # Find position of target item in candidate set (usually index 0)
                target_mask = batch_candidate_tensor == target_id
                if target_mask.any():
                    target_idx_in_candidates = target_mask.nonzero(as_tuple=True)[0][0]
                    relevance_scores[target_idx_in_candidates] = 1.0

                # Check whether proxy labels are in the candidate set and assign relevance
                if len(proxy_ids) > 0:
                    for proxy_id, proxy_sim in zip(proxy_ids, proxy_sims):
                        proxy_mask = batch_candidate_tensor == proxy_id.item()
                        if proxy_mask.any():
                            proxy_idx_in_candidates = proxy_mask.nonzero(as_tuple=True)[0][0]
                            relevance_scores[proxy_idx_in_candidates] = proxy_sim.item()

                # Extract Top-K based on predicted scores
                pred_scores = predicted_scores[i]  # [num_candidates]
                top_k_pred_scores, top_k_pred_indices = torch.topk(pred_scores, k=min(self.k, len(pred_scores)))

                # Extract relevance from Top-K predictions
                predicted_relevance = relevance_scores[top_k_pred_indices]  # [k]

                # Compute DCG
                dcg = calculate_dcg(predicted_relevance.unsqueeze(0), k=self.k)[0]

                # Compute IDCG
                ideal_relevance, _ = torch.sort(relevance_scores, descending=True)
                ideal_relevance = ideal_relevance[:self.k]
                idcg = calculate_dcg(ideal_relevance.unsqueeze(0), k=self.k)[0]

                # Compute NDCG
                if idcg > 0:
                    ndcg_rewards[i] = dcg / (idcg + 1e-10)
                else:
                    ndcg_rewards[i] = 0.0
        
        return ndcg_rewards
    
    def _compute_target_embedding_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward based on similarity with target embedding.

        Higher reward when the generated text is more similar to the target embedding
        and more similar to the target than to the negative embeddings.

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs

        Returns:
            rewards: [batch_size] target embedding similarity reward
                    (target similarity - mean negative similarity)
        """
        # L2 normalization (for cosine similarity)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        # Compute similarity with target embedding
        target_embs = self.target_embeddings[user_ids]  # [batch_size, emb_dim]
        target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)

        target_similarities = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]

        if self.use_full_item_pool:
            # Full item pool: use all items except target as negatives
            # Use average similarity across all item embeddings for memory efficiency
            all_item_embs = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)  # [num_items, emb_dim]

            # Target item IDs
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]

            # Compute similarity with all items
            all_similarities = torch.mm(query_embeddings, all_item_embs.T)  # [batch_size, num_items]

            # Compute mean similarity excluding target
            batch_size = all_similarities.size(0)
            num_items = all_similarities.size(1)

            # Build target mask
            mask = torch.ones(batch_size, num_items, device=self.device, dtype=torch.bool)
            mask[torch.arange(batch_size, device=self.device), target_item_ids] = False

            # Mean similarity of negatives (excluding target)
            negative_mean_similarities = all_similarities[mask].view(batch_size, -1).mean(dim=1)
            rewards = target_similarities - torch.clamp(negative_mean_similarities, min=0.0)
        else:
            # # Compute mean negative similarity (original logic)
            # batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            # negative_ids = batch_candidate_tensor[:, 1:]  # [batch_size, k-1] (exclude first = target)

            # # Compute similarity with all target embeddings
            # all_similarities = torch.mm(query_embeddings, self.target_embeddings.T)  # [batch_size, num_users]

            # # Compute mean similarity
            # negative_mean_similarities = all_similarities.mean(dim=1)  # [batch_size]
            rewards = target_similarities
                
        return rewards
    
    def _compute_infonce_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE (contrastive learning) reward.

        Convert InfoNCE loss to a reward:
        reward = log(exp(sim(q, pos)/tau) / (exp(sim(q, pos)/tau) + sum(exp(sim(q, neg_i)/tau))))

        Encourages higher similarity with target and lower similarity with negatives.

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs

        Returns:
            rewards: [batch_size] InfoNCE reward (higher is better)
        """
        # L2 normalization (for cosine similarity)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        if self.use_full_item_pool:
            # Use full item pool
            # Target item IDs
            target_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]

            # Use InfoNCE embeddings
            target_embs = self.infonce_item_embeddings[target_ids]  # [batch_size, emb_dim]
            target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)

            all_item_embs = torch.nn.functional.normalize(self.infonce_item_embeddings, p=2, dim=1)  # [num_items, emb_dim]

            # 1. Compute similarity with target
            pos_sim = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]
            pos_sim = pos_sim / self.infonce_temperature

            # 2. Compute similarity with all items (excluding target)
            all_sims = torch.mm(query_embeddings, all_item_embs.T)  # [batch_size, num_items]
            all_sims = all_sims / self.infonce_temperature

            # 3. Compute InfoNCE
            # log(exp(pos_sim) / sum(exp(all_sims)))
            log_sum_exp = torch.logsumexp(all_sims, dim=1)  # [batch_size]
            infonce_rewards = pos_sim - log_sum_exp  # [batch_size]
        else:
            # Retrieve candidate tensor (original logic)
            batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            target_ids = batch_candidate_tensor[:, 0]  # [batch_size] - target
            negative_ids = batch_candidate_tensor[:, 1:]  # [batch_size, k-1] - negatives

            # Use InfoNCE embeddings
            target_embs = self.infonce_item_embeddings[target_ids]  # [batch_size, emb_dim]
            target_embs = torch.nn.functional.normalize(target_embs, p=2, dim=1)

            negative_embs = self.infonce_item_embeddings[negative_ids]  # [batch_size, k-1, emb_dim]
            negative_embs = torch.nn.functional.normalize(negative_embs, p=2, dim=2)

            # 1. Compute similarity with target
            pos_sim = (query_embeddings * target_embs).sum(dim=1)  # [batch_size]
            pos_sim = pos_sim / self.infonce_temperature

            # 2. Compute similarity with negatives
            neg_sims = torch.bmm(
                query_embeddings.unsqueeze(1),  # [batch_size, 1, emb_dim]
                negative_embs.transpose(1, 2)   # [batch_size, emb_dim, k-1]
            ).squeeze(1)  # [batch_size, k-1]
            neg_sims = neg_sims / self.infonce_temperature

            # 3. Compute InfoNCE
            # log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
            # = pos_sim - log(exp(pos_sim) + sum(exp(neg_sims)))
            # = pos_sim - logsumexp([pos_sim, neg_sims])
            
            all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # [batch_size, k]
            log_sum_exp = torch.logsumexp(all_sims, dim=1)  # [batch_size]
            
            infonce_rewards = pos_sim - log_sum_exp  # [batch_size]
        
        return infonce_rewards
    
    def _compute_anchor_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        current_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Anchor-Guided GRPO reward.
        Computes reward based on similarity with the last item embedding.
        Dynamic radius control: narrow radius early in training, wider radius later.

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs
            current_step: current training step (if None, use midpoint value)

        Returns:
            rewards: [batch_size] anchor reward
                    soft mode: similarity (similarity reward regardless of radius)
                    hard mode: similarity if in radius, else penalty_value
        """
        # L2 normalization (for cosine similarity)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        # Retrieve last item embeddings
        last_item_embs = self.last_item_embeddings[user_ids]  # [batch_size, emb_dim]
        last_item_embs = torch.nn.functional.normalize(last_item_embs, p=2, dim=1)

        # Compute cosine similarity (similarity ∈ [-1, 1])
        similarities = (query_embeddings * last_item_embs).sum(dim=1)  # [batch_size]

        # # Dynamic radius computation (curriculum learning)
        # if current_step is not None:
        #     # Linearly increase radius as training progresses
        #     progress = min(1.0, current_step / max(1, self.max_steps))
        #     current_radius = self.anchor_radius_start + progress * (self.anchor_radius_end - self.anchor_radius_start)
        # else:
        #     # Use midpoint value if step information is unavailable
        current_radius = (self.anchor_radius_start + self.anchor_radius_end) / 2.0

        if self.anchor_penalty_mode == "soft":
            # Soft mode: use similarity directly as reward
            # Higher similarity yields higher reward regardless of radius
            rewards = similarities
        elif self.anchor_penalty_mode == "hard":
            # Hard mode: similarity reward if within radius, else penalty
            # Use current_radius as threshold
            in_radius = similarities >= current_radius  # [batch_size] boolean
            rewards = torch.where(in_radius, similarities, torch.tensor(self.anchor_penalty_value, device=self.device))
        else:
            raise ValueError(f"Unknown anchor_penalty_mode: {self.anchor_penalty_mode}")
        
        return rewards
    
    def _compute_adaptive_threshold_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Adaptive Threshold Reward.

        Strategy: use the mean similarity with past purchased items (S_base) as a dynamic threshold.

        Formula:
            S_base = mean(CosSim(query, history_items))
            threshold = max(tau_min, S_base)
            R = 1 if CosSim(query, target) > threshold else 0

        Meaning: "The query must be more similar to the correct answer than to past purchases."

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs

        Returns:
            rewards: [batch_size] adaptive threshold reward (0 or 1)
        """
        batch_size = len(user_ids)
        rewards = torch.zeros(batch_size, device=self.device)

        # L2 normalization (for cosine similarity)
        query_embeddings_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        # Retrieve target item IDs
        if self.use_full_item_pool:
            target_item_ids = torch.tensor(
                [self.uid_2_target[uid] for uid in user_ids],
                device=self.device
            )  # [batch_size]
        else:
            batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
            target_item_ids = batch_candidate_tensor[:, 0]  # [batch_size]

        # Retrieve target item embeddings
        target_item_embs = self.item_embeddings[target_item_ids]  # [batch_size, emb_dim]
        target_item_embs_norm = torch.nn.functional.normalize(target_item_embs, p=2, dim=1)

        # Compute similarity between query and target item
        query_target_similarity = (query_embeddings_norm * target_item_embs_norm).sum(dim=1)  # [batch_size]

        for i, uid in enumerate(user_ids):
            uid_item = uid.item() if isinstance(uid, torch.Tensor) else uid

            # 1. Retrieve embeddings of past purchased items
            if uid_item not in self.user_history_items:
                # Use tau_min as threshold if history is empty
                threshold = self.adaptive_tau_min
            else:
                history_item_ids = self.user_history_items[uid_item]  # [history_len]

                # 2. Retrieve history item embeddings
                history_item_embs = self.item_embeddings[history_item_ids]  # [history_len, emb_dim]
                history_item_embs_norm = torch.nn.functional.normalize(history_item_embs, p=2, dim=1)

                # 3. Compute mean similarity between query and history items (S_base)
                query_history_similarities = torch.mm(
                    query_embeddings_norm[i].unsqueeze(0),  # [1, emb_dim]
                    history_item_embs_norm.T  # [emb_dim, history_len]
                ).squeeze(0)  # [history_len]

                s_base = query_history_similarities.mean().item()

                # 4. Dynamic threshold = max(tau_min, S_base)
                threshold = max(self.adaptive_tau_min, s_base)

            # 5. Reward proportional to how much query-target similarity exceeds threshold
            rewards[i] = max(0, query_target_similarity[i].item() - threshold)
        
        return rewards
    
    def _compute_history_proxy_threshold_reward(
        self,
        query_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute History Proxy Threshold Reward.

        Strategy: use the history item most similar to the target (proxy) and set
                  the mean similarity with the remaining history items as the threshold.

        Formula:
            proxy = argmax_i CosSim(target, history_i)
            other_history = history - {proxy}
            S_threshold = mean(CosSim(query, other_history))
            R = max(0, CosSim(query, proxy) - S_threshold)

        Meaning: "Higher reward when the query is closer to the proxy (most target-similar history item)
                  than to the mean of other history items."

        Args:
            query_embeddings: [batch_size, emb_dim] query embeddings
            user_ids: [batch_size] user IDs

        Returns:
            rewards: [batch_size] history proxy threshold reward
        """
        batch_size = len(user_ids)
        rewards = torch.zeros(batch_size, device=self.device)

        # L2 normalization (for cosine similarity)
        query_embeddings_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

        for i, uid in enumerate(user_ids):
            uid_item = uid.item() if isinstance(uid, torch.Tensor) else uid

            # 1. Retrieve proxy item ID (pre-computed)
            if uid_item not in self.user_history_proxy_items:
                # Reward 0 if no proxy
                continue

            proxy_item_id = self.user_history_proxy_items[uid_item]
            history_item_ids = self.user_history_items[uid_item]  # [history_len]

            # 2. Retrieve proxy item embedding
            proxy_item_emb = self.item_embeddings[proxy_item_id]  # [emb_dim]
            proxy_item_emb_norm = torch.nn.functional.normalize(proxy_item_emb.unsqueeze(0), p=2, dim=1)

            # 3. Compute similarity between query and proxy item
            query_proxy_similarity = (query_embeddings_norm[i] * proxy_item_emb_norm.squeeze(0)).sum().item()

            # 4. Compute mean similarity with remaining history items (threshold)
            # History items excluding the proxy
            other_history_mask = history_item_ids != proxy_item_id
            other_history_item_ids = history_item_ids[other_history_mask]

            if len(other_history_item_ids) > 0:
                # Remaining history item embeddings
                other_history_embs = self.item_embeddings[other_history_item_ids]  # [other_len, emb_dim]
                other_history_embs_norm = torch.nn.functional.normalize(other_history_embs, p=2, dim=1)

                # Mean similarity between query and remaining history items
                query_other_similarities = torch.mm(
                    query_embeddings_norm[i].unsqueeze(0),  # [1, emb_dim]
                    other_history_embs_norm.T  # [emb_dim, other_len]
                ).squeeze(0)  # [other_len]

                s_threshold = query_other_similarities.mean().item()
            else:
                # Threshold 0 when history has only the proxy item
                s_threshold = 0.0

            # 5. Reward proportional to how much query-proxy similarity exceeds threshold
            rewards[i] = max(0, query_proxy_similarity - s_threshold)
        
        return rewards
    
    def __call__(
        self,
        generated_texts: List[str],
        user_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        TRL-compatible reward function.

        Args:
            generated_texts: [batch_size] generated texts
            user_ids: [batch_size] user IDs (required)
            **kwargs: additional parameters (targets, histories, trainer_state, etc.)

        Returns:
            rewards: [batch_size] reward values

            If proxy_label_reward=True:
                rewards = base_reward + proxy_label_coef × proxy_label_ndcg
                (Items similar to the target are also partially treated as correct)

            If novelty_reward=True and novelty_annealing=False:
                rewards = novelty_coef × (NDCG × popularity_weight)
                (Higher reward for predicting unpopular items at higher ranks)

            If novelty_reward=True and novelty_annealing=True:
                novelty_ratio = current_step / max_steps (linearly increases 0 → 1)
                rewards = (1 - novelty_ratio) * base_reward + novelty_ratio * novelty_coef * novelty_reward

            Else:
                rewards = base_reward (NDCG/Hit/MRR, etc.)
        """
        # Encode generated texts to embeddings (done only once)
        query_embeddings = self._encode_texts(generated_texts)

        # Compute rank-based rewards
        # Compute ranks (target + negatives)
        ranks, _ = self._compute_similarity_scores(query_embeddings, user_ids, return_scores=False)

        # Compute based on base reward type
        if self.reward_type == "ndcg":
            base_rewards = calculate_ndcg_from_rank(ranks, k=self.k)
        elif self.reward_type == "hit":
            base_rewards = calculate_hit_from_rank(ranks, k=self.k)
        elif self.reward_type == "mrr":
            base_rewards = calculate_mrr_from_rank(ranks, k=self.k)
        elif self.reward_type == "mixed":
            ndcg = calculate_ndcg_from_rank(ranks, k=self.k)
            hit = calculate_hit_from_rank(ranks, k=self.k)
            base_rewards = 0.7 * ndcg + 0.3 * hit
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Save base rewards for wandb logging
        self.last_base_rewards = base_rewards.detach().cpu()

        # Branch based on whether proxy label reward is used
        if self.proxy_label_reward:
            # When using proxy label reward: base_reward + proxy_label_reward
            # Predicted scores also need to be computed
            _, predicted_scores = self._compute_similarity_scores(query_embeddings, user_ids, return_scores=True)
            # Compute proxy label NDCG
            proxy_label_rewards = self._compute_proxy_label_ndcg(query_embeddings, user_ids, predicted_scores)

            # Save for wandb logging
            self.last_proxy_label_rewards = proxy_label_rewards.detach().cpu()

            # Combine the two rewards
            base_rewards = base_rewards + self.proxy_label_coef * proxy_label_rewards
        else:
            self.last_proxy_label_rewards = None

        # Branch based on whether novelty reward is used
        if self.novelty_reward and self.item_popularity_weights is not None:
            # Compute novelty reward
            if self.use_full_item_pool:
                # Full item pool: get directly from uid_2_target
                target_item_ids = torch.tensor(
                    [self.uid_2_target[uid] for uid in user_ids],
                    device=self.device
                )  # [batch_size]
            else:
                # Negative pool: get from candidate_tensor
                batch_candidate_tensor = self.candidate_tensor[user_ids]  # [batch_size, k]
                target_item_ids = batch_candidate_tensor[:, 0]  # [batch_size] - target is always first

            # Popularity weight of target item
            item_weights = self.item_popularity_weights[target_item_ids]  # [batch_size]

            # Novelty = NDCG × popularity_weight
            novelty_rewards = calculate_novelty_ndcg(
                ranks,
                item_weights=item_weights,
                k=self.k,
            )

            # Branch based on whether novelty annealing is applied
            if self.novelty_annealing:
                # Retrieve current step from trainer_state
                trainer_state = kwargs.get("trainer_state", None)

                if trainer_state is not None and hasattr(trainer_state, "global_step"):
                    current_step = trainer_state.global_step
                    # Novelty ratio: linearly increases from 0 (early) to 1 (late)
                    novelty_ratio = min(1.0, current_step / max(1, self.max_steps))
                else:
                    # Default to 0.5 (midpoint) if trainer_state is unavailable
                    novelty_ratio = 0.5

                # Linear interpolation: (1-ratio)*base + ratio*novelty, apply novelty_coef
                rewards = (1.0 - novelty_ratio) * base_rewards + novelty_ratio * self.novelty_coef * novelty_rewards
            else:
                # Use only novelty reward without annealing, apply novelty_coef
                rewards = self.novelty_coef * novelty_rewards
        else:
            # Use base reward
            rewards = base_rewards

        # Add target embedding similarity reward
        if self.target_emb_reward and self.target_embeddings is not None:
            target_emb_rewards = self._compute_target_embedding_reward(query_embeddings, user_ids)
            self.last_target_emb_rewards = target_emb_rewards.detach().cpu()
            rewards = rewards + self.target_emb_coef * target_emb_rewards
        else:
            self.last_target_emb_rewards = None

        # Add InfoNCE reward
        if self.infonce_reward and self.infonce_item_embeddings is not None:
            infonce_rewards = self._compute_infonce_reward(query_embeddings, user_ids)
            self.last_infonce_rewards = infonce_rewards.detach().cpu()
            rewards = rewards + self.infonce_coef * infonce_rewards
        else:
            self.last_infonce_rewards = None

        # Add Anchor-Guided GRPO reward
        if self.anchor_reward and self.last_item_embeddings is not None:
            # Retrieve current step from trainer_state
            trainer_state = kwargs.get("trainer_state", None)
            current_step = None
            if trainer_state is not None and hasattr(trainer_state, "global_step"):
                current_step = trainer_state.global_step

            anchor_rewards = self._compute_anchor_reward(query_embeddings, user_ids, current_step)
            self.last_anchor_rewards = anchor_rewards.detach().cpu()
            rewards = rewards + self.anchor_coef * anchor_rewards
        else:
            self.last_anchor_rewards = None

        # Add Adaptive Threshold reward
        if self.adaptive_threshold_reward and self.user_history_items is not None:
            adaptive_threshold_rewards = self._compute_adaptive_threshold_reward(query_embeddings, user_ids)
            self.last_adaptive_threshold_rewards = adaptive_threshold_rewards.detach().cpu()
            rewards = rewards + self.adaptive_threshold_coef * adaptive_threshold_rewards
        else:
            self.last_adaptive_threshold_rewards = None

        # Add History Proxy Threshold reward
        if self.history_proxy_threshold_reward and self.user_history_proxy_items is not None:
            history_proxy_threshold_rewards = self._compute_history_proxy_threshold_reward(query_embeddings, user_ids)
            self.last_history_proxy_threshold_rewards = history_proxy_threshold_rewards.detach().cpu()
            rewards = rewards + self.history_proxy_threshold_coef * history_proxy_threshold_rewards
        else:
            self.last_history_proxy_threshold_rewards = None

        # Normalize (optional)
        if self.normalize and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return rewards
    
    def get_reward_breakdown(self) -> Dict[str, torch.Tensor]:
        """
        Return the components of the last computed reward.
        Used for wandb logging, etc.

        Returns:
            Dict[str, torch.Tensor]: reward components
                - "base_reward": base reward (NDCG/Hit/MRR)
                - "proxy_label_reward": Proxy label reward (if used)
                - "target_emb_reward": Target embedding similarity reward (if used)
                - "infonce_reward": InfoNCE reward (if used)
                - "anchor_reward": Anchor-Guided GRPO reward (if used)
                - "adaptive_threshold_reward": Adaptive Threshold reward (if used)
                - "history_proxy_threshold_reward": History Proxy Threshold reward (if used)
        """
        breakdown = {}
        
        if self.last_base_rewards is not None:
            breakdown["base_reward"] = self.last_base_rewards
        
        if self.last_proxy_label_rewards is not None:
            breakdown["proxy_label_reward"] = self.last_proxy_label_rewards
        
        if self.last_target_emb_rewards is not None:
            breakdown["target_emb_reward"] = self.last_target_emb_rewards
        
        if self.last_infonce_rewards is not None:
            breakdown["infonce_reward"] = self.last_infonce_rewards
        
        if self.last_anchor_rewards is not None:
            breakdown["anchor_reward"] = self.last_anchor_rewards
        
        if self.last_adaptive_threshold_rewards is not None:
            breakdown["adaptive_threshold_reward"] = self.last_adaptive_threshold_rewards
        
        if self.last_history_proxy_threshold_rewards is not None:
            breakdown["history_proxy_threshold_reward"] = self.last_history_proxy_threshold_rewards
        
        return breakdown
