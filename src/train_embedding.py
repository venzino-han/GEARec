from typing import Dict

import numpy as np
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import random
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim

from utils.common import (
    calculate_hit_rate,
    calculate_ndcg,
    )
from utils.reward_function import (
    calculate_ndcg as calculate_ndcg_batch,
    calculate_hit_rate as calculate_hit_rate_batch,
)

from copy import deepcopy

def parse_arguments():
    """
    Parse command-line arguments for inference and training settings.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a transformer encoder model.")

    # General settings
    parser.add_argument("--run_name", type=str, default="rec_r1")
    parser.add_argument("--data_name", type=str, default="toys")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--check_interval", type=int, default=2000)
    
    # Projection Layer settings
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--beta", type=str, default="0.2")

    parser.add_argument("--days", type=int, default=60, help="Number of days to consider for history.")

    parser.add_argument("--item_target_names_str", type=str, default="review_description")
    parser.add_argument("--use_item_meta", action="store_true", help="Use item meta information.")
    parser.add_argument("--add_item_meta", action="store_true", help="Add item meta information to the embeddings.")
    
    parser.add_argument("--history_limit", type=int, default=8, help="Limit the number of history items to use.")
    parser.add_argument("--num_neg_samples", type=int, default=99, help="Number of negative samples to use.")

    parser.add_argument("--model_name", type=str, default="Llama-3.2-3B-Instruct", help="Name of the model to use.")
    parser.add_argument("--item_model_name", type=str, default="gemma-3-4b-it", help="Name of the model to use.")
    parser.add_argument("--embedding_model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1", help="Name of the model to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--zero_shot", action="store_true", help="Zero-shot mode.")
    parser.add_argument("--filter_top_k", type=int, default=None, help="Filter top-k hit cases for training.")
    parser.add_argument("--filter_rank_csv", type=str, default=None, help="CSV path (e.g. similarity_retrieval_60d_beauty_train_day60.csv). If set, only include train samples with rank <= filter_rank_max in this file.")
    parser.add_argument("--filter_rank_max", type=int, default=None, help="Max rank (inclusive) when using --filter_rank_csv. Samples with rank <= this value are kept in the training set.")

    parser.add_argument("--train_user_embedding_file", type=str, default=None, help="Path to train user embeddings .pt (overrides default path).")
    parser.add_argument("--val_user_embedding_file", type=str, default=None, help="Path to validation user embeddings .pt (overrides default path).")
    parser.add_argument("--test_user_embedding_file", type=str, default=None, help="Path to test user embeddings .pt (overrides default path).")
    parser.add_argument("--item_embedding_file", type=str, default=None, help="Path to item embeddings .pt (overrides default path).")
    parser.add_argument("--crm_name", type=str, default="SASRec", help="Name of the CR model to use.")
    parser.add_argument("--crm_top_k", type=int, default=2000, help="Top-k items to use for training.")
    parser.add_argument("--use_crm", action="store_true", help="Use CRM embeddings.")
    parser.add_argument("--add_crm_emb", action="store_true", help="Add CRM embeddings to the input embeddings.")
    parser.add_argument("--use_pre_train", action="store_true", help="Use pre-train embeddings.")
    parser.add_argument("--use_pref", action="store_true", help="Use pref embeddings.")
    parser.add_argument("--use_entropy", action="store_true", help="Use entropy loss.")
    parser.add_argument("--use_variance", action="store_true", help="Use variance loss.")
    parser.add_argument("--use_uniformity", action="store_true", help="Use uniformity loss.")
    parser.add_argument("--train_data_ratio", type=float, default=1.0, help="Ratio of training data to use (0 < ratio <= 1.0). If < 1.0, randomly sample that fraction of training data.")
    parser.add_argument("--add_anchor_emb", action="store_true", help="Add anchor embedding: use item in 60-day history most similar to user_emb (or last item if none); project anchor and add to projected user_emb.")
    # parser.add_argument("--sasrec_top_k", type=int, default=100, help="Top-k items to use for training.")

    args = parser.parse_args()
    args.beta = float(args.beta)
    args.embedding_model_name_dir = args.embedding_model_name.split("/")[-1]

    # if args.add_item_meta:
    #     args.embedding_dim = args.embedding_dim * 2

    for key, value in vars(args).items():
        print(f"{key:<26}: {value}")

    return args

DTYPE = torch.float32


def entropy_loss(logits, temperature=1.0):
    """
    Computes the entropy loss for a batch of logits.
    
    Args:
        logits (torch.Tensor): The output logits from the model, shape [B, C].
        temperature (float): A temperature parameter to control the sharpness of the distribution.
    
    Returns:
        torch.Tensor: A scalar tensor representing the mean entropy loss for the batch.
    """
    # Apply softmax to get the probability distribution
    probabilities = F.softmax(logits / temperature, dim=-1)
    
    # Compute the entropy: -sum(p * log(p))
    # We add a small value (1e-12) to probabilities to avoid log(0)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=-1)
    
    # Return the average entropy across the batch
    return torch.mean(entropy)

def variance_loss(embeddings):
    """
    Computes the variance loss to prevent model collapse.
    
    Args:
        embeddings (torch.Tensor): A batch of embeddings, shape [N, D].
    
    Returns:
        torch.Tensor: A scalar tensor representing the variance loss.
    """
    # Normalize embeddings to a unit sphere
    normalized_embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute the mean of the embeddings
    mean_embedding = torch.mean(normalized_embeddings, dim=0, keepdim=True)
    
    # Calculate the variance as the mean squared distance from the batch mean
    # The goal is to maximize this variance, so we use a negative sign
    loss = -torch.mean(torch.sum((normalized_embeddings - mean_embedding) ** 2, dim=1))
    
    return loss

def uniformity_loss(embeddings, t=2):
    """
    Computes the uniformity loss to encourage embeddings to be uniformly distributed on the hypersphere.
    
    Args:
        embeddings (torch.Tensor): A batch of embeddings, shape [N, D].
        t (float): A temperature parameter for the loss function.
    
    Returns:
        torch.Tensor: A scalar tensor representing the uniformity loss.
    """
    # Normalize embeddings
    normalized_embeddings = F.normalize(embeddings, dim=-1)
    
    # Compute pairwise squared Euclidean distances
    pairwise_distances_sq = torch.cdist(normalized_embeddings, normalized_embeddings, p=2) ** 2
    
    # Compute the negative exponential of the distances
    loss = torch.log(torch.exp(-t * pairwise_distances_sq).sum() / (len(embeddings) ** 2))
    
    return loss


def calculate_item_novelty_scores(train_user_seq_data: Dict, test_user_target: Dict):
    """
    Calculate item novelty scores based on item popularity in the training data.
    Novelty score for each item is calculated as -log2(p(i)), where p(i) is the probability of item i.
    
    Args:
        train_user_seq_data (Dict): Dictionary mapping user_id to list of item_ids in their history.
        test_user_target (Dict): Dictionary mapping user_id to their target item_id.
    
    Returns:
        Tuple[Dict, int]: Tuple of (item_novelty_scores, total_interactions)
            - item_novelty_scores: Dictionary mapping item_id to novelty score
            - total_interactions: Total number of interactions in the training set
    """
    # Calculate item popularity from training data
    item_popularity = {}
    total_interactions = 0
    for user_id, item_seq in train_user_seq_data.items():
        for item_id in item_seq:
            item_popularity[item_id] = item_popularity.get(item_id, 0) + 1
            total_interactions += 1
    
    # Add train targets to popularity count
    for user_id, target_item in test_user_target.items():
        item_popularity[target_item] = item_popularity.get(target_item, 0) + 1
        total_interactions += 1
    
    print(f"Total interactions in train set: {total_interactions}")
    print(f"Number of unique items in train set: {len(item_popularity)}")
    
    # Calculate novelty score for each item: -log2(p(i))
    item_novelty_scores = {}
    for item_id, count in item_popularity.items():
        p_i = count / total_interactions
        novelty_i = -np.log2(p_i)
        item_novelty_scores[item_id] = novelty_i
    
    return item_novelty_scores, total_interactions


class Projector(nn.Module):
    def __init__(self, args, embbeding_dim=1024, output_dim=512, crm_output_dim=128, dropout=0.1):
        """
        Initializes the Projector model.

        Args:
            args (Namespace): Arguments object containing model parameters like beta.
            embbeding_dim (int): Input embedding dimension.
            output_dim (int): Output dimension after projection.
            dropout (float): Dropout rate for MLP layers (though only applied if MLP layers are more complex).
        """
        super().__init__()
        self.args = args

        self.item_mlp = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(embbeding_dim, output_dim),
            # nn.GELU(),
            # nn.Linear(output_dim, output_dim),
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(embbeding_dim, output_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.anchor_mlp = None
        if getattr(args, "add_anchor_emb", False):
            self.anchor_mlp = nn.Sequential(nn.Linear(embbeding_dim, output_dim))

        # Note: TransformerEncoder, positional embeddings, user_mlp, and meta_mlp
        # were commented out/unused in the original forward pass, so they are removed
        # to keep the class concise and reflective of its current operation.
    def set_zero_shot(self):
        self.user_mlp = nn.Sequential(nn.Identity())
        self.item_mlp = nn.Sequential(nn.Identity())
        if self.anchor_mlp is not None:
            self.anchor_mlp = nn.Sequential(nn.Identity())
        return self

    def infonce_loss(self, query_emb, positive_emb, negatives_emb, temperature=0.07):
        """
        query: [B, D]
        positive: [B, D]
        negatives: [B, K, D]
        """
        query_emb = F.normalize(query_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)
        negatives_emb = F.normalize(negatives_emb, dim=-1)

        pos_sim = torch.sum(query_emb * positive_emb, dim=1, keepdim=True)
        neg_sim = torch.bmm(negatives_emb, query_emb.unsqueeze(2)).squeeze(2)  # [B, K]
        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature

        labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
        return F.cross_entropy(logits, labels), logits

    def forward(self,
        user_seq_emb, pos_item_emb, neg_item_embs, anchor_item_emb=None,
        ):
        """
        Forward pass for the Projector model.
        If anchor_item_emb is provided (add_anchor_emb), query = project(user) + project(anchor).
        """
        user_seq_emb = user_seq_emb.to(torch.float32)
        pos_item_emb = pos_item_emb.to(torch.float32)
        neg_item_embs = neg_item_embs.to(torch.float32)

        x = user_seq_emb
        x = self.item_mlp(x)
        # x = self.user_mlp(x)
        if anchor_item_emb is not None and self.anchor_mlp is not None:
            anchor_item_emb = anchor_item_emb.to(torch.float32)
            x = x + self.anchor_mlp(anchor_item_emb)
        pos_item_emb = self.item_mlp(pos_item_emb)
        neg_item_embs = self.item_mlp(neg_item_embs)

        # Compute the InfoNCE loss
        loss, logits = self.infonce_loss(x, pos_item_emb, neg_item_embs)


        # pos_item_emb_2 = self.crm_item_mlp(pos_item_emb)
        # neg_item_embs_2 = self.crm_item_mlp(neg_item_embs)
        # user_emb_2 = self.crm_item_mlp(x)

        # loss += 0.01 * self.infonce_loss(crm_user_emb, pos_item_emb_2, neg_item_embs_2)
        # loss += 0.01 * self.infonce_loss(user_emb_2, crm_pos_item_emb, crm_neg_item_embs)
        # add mse loss between crm_user_emb and user_emb_2
        # loss += 0.1 * torch.mean(torch.norm(crm_user_emb - user_emb_2, p=2, dim=1))
        # # add mse loss between crm_pos_item_emb and pos_item_emb_2
        # loss += 0.1 * torch.mean(torch.norm(crm_pos_item_emb - pos_item_emb_2, p=2, dim=1))
        # # add mse loss between crm_neg_item_embs and neg_item_embs_2
        # loss += 0.1 * torch.mean(torch.norm(crm_neg_item_embs - neg_item_embs_2, p=2, dim=2))


        # reg term for the projection layers
        if not self.args.zero_shot:
            # reg_loss = torch.mean(torch.norm(self.item_mlp[0].weight.data, p=2, dim=0)) + \
            #             torch.mean(torch.norm(self.item_mlp[0].bias.data, p=2, dim=0))
            #                     # Calculate the regularization term (L2-norm)
            # Apply norm across the embedding dimension for each tensor
            reg_loss = torch.mean(torch.norm(x, p=2, dim=1)) + \
                        torch.mean(torch.norm(pos_item_emb, p=2, dim=1)) + \
                        torch.mean(torch.norm(neg_item_embs, p=2, dim=2)) # dim=2 for [B, K, D_out]
            loss += self.args.beta * reg_loss
            if args.use_entropy:
                loss += self.args.beta * entropy_loss(logits)
            if args.use_variance:
                loss += self.args.beta * variance_loss(x)
            if args.use_uniformity:
                loss += self.args.beta * uniformity_loss(x)

                    
        # Add regularization loss to the total loss
        return x, loss
    
    def get_item_emb(self, item_embs):
        """
        item_embs: [B, N, D]
        """
        item_embs = item_embs.to(torch.float32)
        item_embs = self.item_mlp(item_embs)
        return item_embs

def get_dataloader(
            user_embs, pos_item_ids, neg_item_sets, item_embeddings,
            batch_size=128, shuffle=True, filter_top_k=None, data_ratio=1.0,
            filter_rank_csv=None, filter_rank_max=None,
            anchor_item_ids=None,
        ):
    """
    anchor_item_ids: optional list of length N. If set, batch returns (user_embs, anchor_item_embs, pos_item_embs, neg_item_embs).
    """
    pos_item_embs = item_embeddings[pos_item_ids]
    neg_item_embs = item_embeddings[neg_item_sets]
    anchor_item_embs = None
    if anchor_item_ids is not None:
        anchor_item_embs = item_embeddings[anchor_item_ids]

    if filter_rank_csv is not None and filter_rank_max is not None:
        rank_df = pd.read_csv(filter_rank_csv)
        if "rank" not in rank_df.columns:
            raise ValueError(f"CSV {filter_rank_csv} must have a 'rank' column")
        rank_df = rank_df[rank_df["rank"] <= filter_rank_max]
        valid_user_ids = set(rank_df["user_id"].astype(int).tolist())
        indices = [i for i in range(user_embs.shape[0]) if (i + 1) in valid_user_ids]
        user_embs = user_embs[indices]
        pos_item_embs = pos_item_embs[indices]
        neg_item_embs = neg_item_embs[indices]
        if anchor_item_embs is not None:
            anchor_item_embs = anchor_item_embs[indices]
        print(f"Filtered by rank CSV: {filter_rank_csv} (rank<={filter_rank_max}) -> {len(indices)} samples")
    elif filter_top_k is not None:
        hit_df = pd.read_csv(f"results/{args.data_name}_{args.run_name}_zero_shot_train_{args.model_name}.csv")
        hit_df = hit_df.query(f"rank <= {filter_top_k}")
        indices = hit_df["user_id"].values.tolist()
        indices = [i - 1 for i in indices]
        user_embs = user_embs[indices]
        pos_item_embs = pos_item_embs[indices]
        neg_item_embs = neg_item_embs[indices]
        if anchor_item_embs is not None:
            anchor_item_embs = anchor_item_embs[indices]

    print("user_embs.shape:", user_embs.shape)
    print("pos_item_embs.shape:", pos_item_embs.shape)
    print("neg_item_embs.shape:", neg_item_embs.shape)
    if anchor_item_embs is not None:
        print("anchor_item_embs.shape:", anchor_item_embs.shape)

    if user_embs.shape[0] != pos_item_embs.shape[0]:
        user_embs = torch.cat([torch.zeros(1, user_embs.shape[-1]), user_embs], dim=0)
        if anchor_item_embs is not None:
            anchor_item_embs = torch.cat([torch.zeros(1, anchor_item_embs.shape[-1]), anchor_item_embs], dim=0)

    assert user_embs.shape[0] == pos_item_embs.shape[0] == neg_item_embs.shape[0]
    assert user_embs.shape[-1] == pos_item_embs.shape[-1] == neg_item_embs.shape[-1]

    if anchor_item_embs is not None:
        dataset = TensorDataset(user_embs, anchor_item_embs, pos_item_embs, neg_item_embs)
    else:
        dataset = TensorDataset(user_embs, pos_item_embs, neg_item_embs)

    if data_ratio < 1.0 and shuffle:
        num_samples = int(len(dataset) * data_ratio)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Number of samples after {data_ratio:.2%} sampling: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return dataloader


def get_sequential_dataloader(user_seq_data, item_embeddings, neg_samples_per_item=99, batch_size=128, shuffle=True):
    """
    연속적인 아이템 간의 유사도를 학습하기 위한 데이터로더를 생성합니다.
    
    Args:
        user_seq_data: dict, {user_id: [item_id1, item_id2, ...]} 형태의 사용자별 아이템 시퀀스
        item_embeddings: torch.Tensor, [M, D] 형태의 아이템 임베딩
        neg_samples_per_item: int, 각 positive pair에 대한 negative sample 개수
        batch_size: int, 배치 크기
        shuffle: bool, 데이터 셔플 여부
    
    Returns:
        DataLoader: (current_item_emb, next_item_emb, neg_item_embs) 형태의 데이터로더
    """
    current_item_ids = []
    next_item_ids = []
    
    # 모든 사용자의 시퀀스에서 연속적인 아이템 쌍을 추출
    for user_id, item_sequence in user_seq_data.items():
        if len(item_sequence) < 2:  # 시퀀스가 너무 짧으면 스킵
            continue
        item_sequence = item_sequence[-4:] # only use the last 8 items
            
        # 연속적인 아이템 쌍 생성 (current -> next)
        for i in range(len(item_sequence) - 1):
            current_item_ids.append(item_sequence[i])
            next_item_ids.append(item_sequence[i + 1])
    
    if len(current_item_ids) == 0:
        raise ValueError("No valid sequential pairs found in the data")
    
    # 텐서로 변환
    current_item_ids = torch.tensor(current_item_ids, dtype=torch.long)
    next_item_ids = torch.tensor(next_item_ids, dtype=torch.long)
    
    # 임베딩 추출
    current_item_embs = item_embeddings[current_item_ids]  # [N, D]
    next_item_embs = item_embeddings[next_item_ids]  # [N, D]
    
    # Negative sampling
    num_items = item_embeddings.shape[0]
    num_pairs = len(current_item_ids)
    
    # 각 positive pair에 대해 negative samples 생성
    neg_item_ids = torch.randint(1, num_items, (num_pairs, neg_samples_per_item), dtype=torch.long)
    
    # positive item과 같은 negative sample이 선택되지 않도록 처리
    for i in range(num_pairs):
        for j in range(neg_samples_per_item):
            while neg_item_ids[i, j] == next_item_ids[i]:
                neg_item_ids[i, j] = torch.randint(1, num_items, (1,), dtype=torch.long)
    
    neg_item_embs = item_embeddings[neg_item_ids]  # [N, K, D]
    current_item_embs = current_item_embs.unsqueeze(1)
    # next_item_embs = next_item_embs.unsqueeze(1)
    
    print(f"Sequential pairs: {num_pairs}")
    print("current_item_embs.shape:", current_item_embs.shape)
    print("next_item_embs.shape:", next_item_embs.shape)
    print("neg_item_embs.shape:", neg_item_embs.shape)
    
    # 차원 확인
    assert current_item_embs.shape[0] == next_item_embs.shape[0] == neg_item_embs.shape[0], "Embedding dimensions do not match"
    
    # TensorDataset과 DataLoader 생성
    dataset = TensorDataset(current_item_embs, next_item_embs, neg_item_embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_rec(
    args, model, train_loader, valid_loader, epochs=10, device="cuda:0"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    model = model.to(device)
    best_model = model

    best_val_loss = float('inf')
    prev_val_loss = float('inf')
    early_stop_count = 0

    use_anchor = getattr(args, "add_anchor_emb", False)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = datetime.now()
        for batch_idx, batch in enumerate(train_loader):
            if use_anchor:
                user_embs, anchor_embs, pos_item_embs, neg_item_embs = batch
                anchor_embs = anchor_embs.to(device)
            else:
                user_embs, pos_item_embs, neg_item_embs = batch
                anchor_embs = None
            user_embs = user_embs.to(device)
            pos_item_embs = pos_item_embs.to(device)
            neg_item_embs = neg_item_embs.to(device)

            optimizer.zero_grad()
            _, loss = model(user_embs, pos_item_embs, neg_item_embs, anchor_item_emb=anchor_embs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        end_time = datetime.now()
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for batch_idx, batch in enumerate(valid_loader):
                if use_anchor:
                    user_embs, anchor_embs, pos_item_embs, neg_item_embs = batch
                    anchor_embs = anchor_embs.to(device)
                else:
                    user_embs, pos_item_embs, neg_item_embs = batch
                    anchor_embs = None
                user_embs = user_embs.to(device)
                pos_item_embs = pos_item_embs.to(device)
                neg_item_embs = neg_item_embs.to(device)

                _, val_loss = model(user_embs, pos_item_embs, neg_item_embs, anchor_item_emb=anchor_embs)
                total_val_loss += val_loss.item()
            val_loss = total_val_loss / len(valid_loader)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch time: {end_time - start_time}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model).to("cpu")
        if val_loss > prev_val_loss:
            early_stop_count += 1
            if early_stop_count >= args.early_stop:
                print("Early stopping triggered.")
                break
        prev_val_loss = val_loss

    return best_model

def test_transformer_encoder(
    model, test_loader, item_embs, device="cuda:0", use_anchor=False,
):
    model = model.to(device)
    model.eval()
    total_user_embs = []
    with torch.no_grad():
        total_loss = 0
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if use_anchor:
                user_embs, anchor_embs, pos_item_embs, neg_item_embs = batch
                anchor_embs = anchor_embs.to(device)
            else:
                user_embs, pos_item_embs, neg_item_embs = batch
                anchor_embs = None
            user_embs = user_embs.to(device)
            pos_item_embs = pos_item_embs.to(device)
            neg_item_embs = neg_item_embs.to(device)

            user_embs, loss = model(user_embs, pos_item_embs, neg_item_embs, anchor_item_emb=anchor_embs)
            total_loss += loss.item()
            total_user_embs.append(user_embs.cpu())
        total_user_embs = torch.cat(total_user_embs, dim=0)
        print("Total Test user embeddings shape:", total_user_embs.shape)
        test_loss = total_loss / len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")

        total_item_embs = model.get_item_emb(item_embs.to(device))

    return test_loss, total_user_embs, total_item_embs


def get_time_aware_user_history(args, split="train"):
    reviews_file = f"data/{args.data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    uid_2_history_text = {}
    if split == "train":
        index = -3
    elif split == "valid":
        index = -2
    elif split == "test":
        index = -1

    for user_id, reviews in user2reviews.items():
        history_iids = []
        target_timestamp = int(reviews[index]["timestamp"])
        for review in reviews[index-args.history_limit:index]:
            item_id = int(review["item_id"])
            timestamp = int(review["timestamp"])
            # only include reviews that are in the last k days
            if target_timestamp - timestamp > args.days * 24 * 60 * 60:
                continue
            history_iids.append(item_id)
        if len(history_iids) == 0:
            history_iids.append(item_id)
        history_iids.reverse()  # reverse the order to make it time-aware
        uid_2_history_text[user_id] = history_iids  # reverse the order to make it time-aware
    print(f"Number of users with history in {split} set: {len(uid_2_history_text)}")
    return uid_2_history_text


def compute_anchor_item_ids(user_embeddings, user_seq_data, history_60d, item_embeddings, device="cpu"):
    """
    For each user, anchor = item in 60-day history most similar to user_emb; if no 60-day history, use last item.
    user_embeddings: [N+1, D] (index 0 unused)
    Returns: list of length N (anchor item id per user_id 1..N)
    """
    user_embeddings = user_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)
    if item_embeddings.dim() == 2:
        item_embeddings = F.normalize(item_embeddings, dim=-1)
    anchor_ids = []
    for user_id in range(1, user_embeddings.shape[0]):
        u_emb = user_embeddings[user_id : user_id + 1]  # [1, D]
        u_emb = F.normalize(u_emb, dim=-1)
        history_60 = history_60d.get(user_id, [])
        if len(history_60) == 0:
            # fallback: last item in full sequence
            seq = user_seq_data.get(user_id, [])
            anchor_id = int(seq[-1]) if seq else 0
        else:
            item_embs = item_embeddings[history_60]  # [L, D]
            sim = torch.matmul(u_emb, item_embs.T).squeeze(0)  # [L]
            best_idx = sim.argmax().item()
            anchor_id = int(history_60[best_idx])
        anchor_ids.append(anchor_id)
    return anchor_ids


def get_user_embeddings(args, user_embeddings, user_item_ids, item_embeddings):
    """
    user_embeddings: [N, D]
    user_item_ids: [N, K]
    item_embeddings: [M, D]
    """
    user_embs = []
    # dtype = torch.float32
    # convert user embeddings to the same dtype as item embeddings
    user_embeddings = user_embeddings.to(DTYPE)
    item_embeddings = item_embeddings.to(DTYPE)
    for user_id in range(1, user_embeddings.shape[0]):
        item_ids = user_item_ids[user_id]    
        if len(item_ids) == 0:
            continue
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[item_ids].to(user_emb.device)
        merged_user_emb = torch.zeros((1, args.embedding_dim), dtype=DTYPE).to(user_emb.device)
        merged_user_emb[0, :] = user_emb
        num_items = len(item_ids)
        # merged_user_emb[1:num_items+1, :] = item_embs
        user_embs.append(merged_user_emb)
    
    return torch.stack(user_embs, dim=0)  # [N, 9, D]


def get_crm_embeddings(args,):
    # load the crm embeddings
    crm_item_embeddings = torch.load(f"sasrec_results/{args.crm_name}_{args.data_name}_item_embeddings.pt").detach().cpu()
    crm_train_user_embeddings = torch.load(f"sasrec_results/{args.crm_name}_{args.data_name}_train_user_embeddings.pt").detach().cpu()
    crm_val_user_embeddings = torch.load(f"sasrec_results/{args.crm_name}_{args.data_name}_val_user_embeddings.pt").detach().cpu()
    crm_test_user_embeddings = torch.load(f"sasrec_results/{args.crm_name}_{args.data_name}_test_user_embeddings.pt").detach().cpu()

    return crm_item_embeddings, crm_train_user_embeddings, crm_val_user_embeddings, crm_test_user_embeddings


def load_item_popularity_novelty(data_name, num_items):
    """
    Load or compute item popularity, novelty, and cold/hot sets (same as evaluator).
    - Cold: 인기도 하위 80% 아이템
    - Hot: 인기도 상위 20% 아이템
    Returns:
        item_popularity: np.ndarray (num_items+1,)
        item_novelty: np.ndarray (num_items+1,)
        cold_items: set
        hot_items: set
    """
    cache_dir = Path(f"./data/{data_name}")
    popularity_file = cache_dir / "item_popularity.npy"
    novelty_file = cache_dir / "item_novelty.npy"
    cold_hot_file = cache_dir / "item_cold_hot.json"

    if popularity_file.exists() and novelty_file.exists() and cold_hot_file.exists():
        try:
            item_popularity = np.load(popularity_file)
            item_novelty = np.load(novelty_file)
            with open(cold_hot_file, "r") as f:
                cold_hot_data = json.load(f)
            cold_items = set(cold_hot_data["cold_items"])
            hot_items = set(cold_hot_data["hot_items"])
            print(f"Loaded item popularity/novelty from cache. Mean novelty: {item_novelty.mean():.4f}")
            print(f"Hot items: {len(hot_items)}, Cold items: {len(cold_items)}")
            return item_popularity, item_novelty, cold_items, hot_items
        except Exception as e:
            print(f"Failed to load cache: {e}. Computing from scratch...")

    sequential_file = Path(f"./data/{data_name}/sequential_data.txt")
    item_counts = np.zeros(num_items + 1, dtype=np.int32)
    with open(sequential_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            items = [int(x) for x in parts[1:-2]]
            for item_id in items:
                item_counts[item_id] += 1
    item_popularity = (item_counts + 1) / item_counts.sum()
    item_novelty = -np.log(item_popularity)
    sorted_counts = np.sort(item_counts)[::-1]
    threshold_idx = int(len(sorted_counts) * 0.2)
    threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else 0
    hot_mask = (item_counts >= threshold) & (np.arange(len(item_counts)) > 0)
    cold_mask = (item_counts < threshold) & (np.arange(len(item_counts)) > 0)
    hot_items = set(np.where(hot_mask)[0].tolist())
    cold_items = set(np.where(cold_mask)[0].tolist())
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(popularity_file, item_popularity)
        np.save(novelty_file, item_novelty)
        with open(cold_hot_file, "w") as f:
            json.dump({"cold_items": list(cold_items), "hot_items": list(hot_items)}, f)
    except Exception as e:
        print(f"Failed to save cache: {e}")
    print(f"Computed item popularity/novelty. Hot: {len(hot_items)}, Cold: {len(cold_items)}")
    return item_popularity, item_novelty, cold_items, hot_items


def identify_cold_warm_indices(targets, hot_items, cold_items):
    """Cold/Hot 타겟 인덱스 구분 (evaluator와 동일)."""
    cold_indices = []
    hot_indices = []
    for i, target in enumerate(targets):
        if target in hot_items:
            hot_indices.append(i)
        else:
            cold_indices.append(i)
    return cold_indices, hot_indices


def compute_retrieval_metrics_embedding(
    query_embeddings,
    item_embeddings,
    targets,
    histories,
    item_novelty,
    cold_items,
    hot_items,
    ks=(5, 10, 20),
    batch_size=512,
    device="cuda:0",
):
    """
    Evaluator의 compute_retrieval_metrics와 동일한 방식으로 메트릭 계산.
    """
    cold_indices, hot_indices = identify_cold_warm_indices(targets, hot_items, cold_items)
    cold_indices_set = set(cold_indices)
    print(f"  Cold targets: {len(cold_indices)}, Hot targets: {len(hot_indices)}")

    metrics = {f"ndcg@{k}": [] for k in ks}
    metrics.update({f"hit@{k}": [] for k in ks})
    metrics.update({f"cold_ndcg@{k}": [] for k in ks})
    metrics.update({f"cold_hit@{k}": [] for k in ks})
    metrics.update({f"hot_ndcg@{k}": [] for k in ks})
    metrics.update({f"hot_hit@{k}": [] for k in ks})
    metrics.update({f"novelty@{k}": [] for k in ks})
    metrics.update({f"novelty_hit@{k}": [] for k in ks})
    rank_info = []

    num_queries = query_embeddings.shape[0]
    num_batches = (num_queries + batch_size - 1) // batch_size
    query_embeddings = query_embeddings.to(device)
    item_embeddings = item_embeddings.to(device)

    for batch_idx in tqdm(range(num_batches), desc="Computing metrics"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_queries)
        batch_query_emb = query_embeddings[start_idx:end_idx]
        batch_targets = [targets[i] for i in range(start_idx, end_idx)]
        batch_histories = [histories[i] for i in range(start_idx, end_idx)]

        batch_scores = torch.matmul(batch_query_emb, item_embeddings.T)

        for i in range(len(batch_targets)):
            history_indices = [idx for idx in batch_histories[i] if idx != batch_targets[i]]
            if history_indices:
                batch_scores[i, history_indices] = -float("inf")

        for i in range(len(batch_targets)):
            global_idx = start_idx + i
            target_item = batch_targets[i]
            target_score = batch_scores[i, target_item].item()
            rank = (batch_scores[i] > target_score).sum().item() + 1
            is_cold = global_idx in cold_indices_set
            k500 = min(500, batch_scores.shape[1])
            top_500 = torch.topk(batch_scores[i], k500)
            rank_info.append({
                "target_item": target_item,
                "target_score": target_score,
                "rank": rank,
                "is_cold": is_cold,
                "target_novelty": float(item_novelty[target_item]),
                "top_500_indices": top_500.indices.cpu().tolist(),
                "top_500_scores": top_500.values.cpu().tolist(),
            })

        for k in ks:
            ndcg_scores = calculate_ndcg_batch(
                batch_scores,
                batch_targets,
                batch_histories,
                k=k,
                use_negatives_only=False,
            )
            hit_scores = calculate_hit_rate_batch(
                batch_scores,
                batch_targets,
                batch_histories,
                k=k,
                use_negatives_only=False,
            )
            metrics[f"ndcg@{k}"].extend(ndcg_scores.cpu().tolist())
            metrics[f"hit@{k}"].extend(hit_scores.cpu().tolist())

            for i in range(len(batch_targets)):
                global_idx = start_idx + i
                is_cold = global_idx in cold_indices_set
                if is_cold:
                    metrics[f"cold_ndcg@{k}"].append(ndcg_scores[i].item())
                    metrics[f"cold_hit@{k}"].append(hit_scores[i].item())
                else:
                    metrics[f"hot_ndcg@{k}"].append(ndcg_scores[i].item())
                    metrics[f"hot_hit@{k}"].append(hit_scores[i].item())

            for i in range(len(batch_targets)):
                target_item = batch_targets[i]
                top_k_items = torch.topk(batch_scores[i], k).indices.cpu().numpy()
                avg_novelty = np.mean([item_novelty[item] for item in top_k_items])
                metrics[f"novelty@{k}"].append(avg_novelty)
                hit_value = hit_scores[i].item()
                novelty_hit = hit_value * item_novelty[target_item]
                metrics[f"novelty_hit@{k}"].append(novelty_hit)

    return metrics, rank_info


if __name__ == "__main__":
    args = parse_arguments()


    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")



    # Load the item embeddings
    if args.item_embedding_file is not None:
        item_embedding_file = args.item_embedding_file
    else:
        item_embedding_file = f"data_emb/{args.data_name}_{args.item_target_names_str}_{args.token_limit}_{args.item_model_name}_{args.embedding_model_name_dir}_emb.pt"
        if "title" in args.item_target_names_str:
            item_embedding_file = f"data_emb/{args.data_name}_{args.item_target_names_str}_{args.embedding_model_name_dir}_emb.pt"
    print("item_embedding_file:", item_embedding_file)
    item_embeddings = torch.load(item_embedding_file).cpu()
    # except:
    #     item_embedding_file = f"data_emb/{args.data_name}_{args.item_target_names_str}_args.embedding_model_name_dir}_emb.pt"

    # normalize the item embeddings

    # item_meta_embeddings = item_embeddings
    

    args.input_embedding_dim = item_embeddings.shape[1]

    # # load item meta embeddings
    # item_meta_embedding_file = f"data_emb/{args.data_name}_title_brand_category_feature_description_{args.embedding_model_name_dir}_emb.pt"
    # item_meta_embeddings = torch.load(item_meta_embedding_file).cpu()



    # load pos_item_ids and neg_item_sets
    data_file = f"data/{args.data_name}/sequential_data.txt"
    train_user_seq_data = {}
    val_user_seq_data = {}
    test_user_seq_data = {}

    pre_train_pos_item_ids = []
    train_pos_item_ids = []
    val_pos_item_ids = []
    test_pos_item_ids = []
    test_user_target = {}
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]
            train_user_seq_data[user_id] = parts[:-3]
            val_user_seq_data[user_id] = parts[:-2]
            test_user_seq_data[user_id] = parts[:-1]
            pre_train_pos_item_ids.append(parts[-4])
            train_pos_item_ids.append(parts[-3])
            val_pos_item_ids.append(parts[-2])
            test_pos_item_ids.append(parts[-1])
            test_user_target[user_id] = parts[-1]
    # load the user embeddings

    
    train_user_embeddings = torch.zeros((len(train_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)
    val_user_embeddings = torch.zeros((len(val_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)
    test_user_embeddings = torch.zeros((len(test_user_seq_data)+1, args.embedding_dim), dtype=torch.float32)
    if args.run_name == "last_title":
        # for ablation study 
        for user_id, test_item_ids in test_user_seq_data.items():
            test_last_item_id = test_item_ids[-1]
            train_last_item_id = train_user_seq_data[user_id][-1]
            val_last_item_id = val_user_seq_data[user_id][-1]
            train_user_embeddings[user_id] = item_embeddings[train_last_item_id]
            val_user_embeddings[user_id] = item_embeddings[val_last_item_id]
            test_user_embeddings[user_id] = item_embeddings[test_last_item_id]
        train_user_embeddings = train_user_embeddings.unsqueeze(1)
        val_user_embeddings = val_user_embeddings.unsqueeze(1)
        test_user_embeddings = test_user_embeddings.unsqueeze(1)
    elif args.run_name == "avg_title":
        # for ablation study 
        for user_id, test_item_ids in test_user_seq_data.items():
            train_item_ids = train_user_seq_data[user_id]
            val_item_ids = val_user_seq_data[user_id]
            test_item_ids = test_item_ids
            train_user_embeddings[user_id] = item_embeddings[train_item_ids].mean(dim=0)
            val_user_embeddings[user_id] = item_embeddings[val_item_ids].mean(dim=0)
            test_user_embeddings[user_id] = item_embeddings[test_item_ids].mean(dim=0)
        train_user_embeddings = train_user_embeddings.unsqueeze(1)
        val_user_embeddings = val_user_embeddings.unsqueeze(1)
        test_user_embeddings = test_user_embeddings.unsqueeze(1)
    else:
        if args.train_user_embedding_file is not None:
            train_user_embedding_file = args.train_user_embedding_file
            train_user_embeddings = torch.load(train_user_embedding_file).to("cpu")
        else:
            try:
                train_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_encoder_emb.pt"
                train_user_embeddings = torch.load(train_user_embedding_file).to("cpu")
            except Exception:
                train_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_pred_emb.pt"
                train_user_embeddings = torch.load(train_user_embedding_file).to("cpu")
        if args.val_user_embedding_file is not None:
            val_user_embedding_file = args.val_user_embedding_file
            val_user_embeddings = torch.load(val_user_embedding_file).to("cpu")
        else:
            try:
                val_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_encoder_emb.pt"
                val_user_embeddings = torch.load(val_user_embedding_file).to("cpu")
            except Exception:
                val_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_valid_pred_emb.pt"
                val_user_embeddings = torch.load(val_user_embedding_file).to("cpu")
        if args.test_user_embedding_file is not None:
            test_user_embedding_file = args.test_user_embedding_file
            test_user_embeddings = torch.load(test_user_embedding_file).to("cpu")
        else:
            try:
                test_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_encoder_emb.pt"
                test_user_embeddings = torch.load(test_user_embedding_file).to("cpu")
            except Exception:
                test_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_test_pred_emb.pt"
                test_user_embeddings = torch.load(test_user_embedding_file).to("cpu")

        # Get additional user embeddings
        if args.use_pref:
            train_user_embedding_file = f"data_emb/pref_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_train_pred_emb.pt"
            train_pref_user_embeddings = torch.load(train_user_embedding_file).to("cpu").to(DTYPE)
            
        if args.use_pre_train:
            pre_train_user_embedding_file = f"data_emb/{args.run_name}_{args.token_limit}_user_preference_{args.data_name}_{args.model_name}_{args.embedding_model_name_dir}_pre_train_pred_emb.pt"
            pre_train_user_embeddings = torch.load(pre_train_user_embedding_file).to("cpu").to(DTYPE)



    if args.use_crm:
        print("Using CRM embeddings")
        crm_item_embeddings, crm_train_user_embeddings, crm_val_user_embeddings, crm_test_user_embeddings = get_crm_embeddings(args)
        crm_item_embeddings = crm_item_embeddings.to("cpu")
        crm_train_user_embeddings = crm_train_user_embeddings.to("cpu")
        crm_val_user_embeddings = crm_val_user_embeddings.to("cpu")
        crm_test_user_embeddings = crm_test_user_embeddings.to("cpu")
        train_user_embeddings = torch.cat([train_user_embeddings, crm_train_user_embeddings], dim=1)
        val_user_embeddings = torch.cat([val_user_embeddings, crm_val_user_embeddings], dim=1)
        test_user_embeddings = torch.cat([test_user_embeddings, crm_test_user_embeddings], dim=1)
        item_embeddings = torch.cat([item_embeddings, crm_item_embeddings], dim=1)
        args.input_embedding_dim += crm_item_embeddings.shape[1]

    model = Projector(
        args=args,
        embbeding_dim=args.input_embedding_dim,
        output_dim=args.embedding_dim,
        dropout=args.dropout,
    )
        
    print("train_user_embeddings.shape:", train_user_embeddings.shape)
    print("val_user_embeddings.shape:", val_user_embeddings.shape)
    print("test_user_embeddings.shape:", test_user_embeddings.shape)
    print("dtype of train_user_embeddings:", train_user_embeddings.dtype)

    print("item_embeddings.shape:", item_embeddings.shape)

    # to dtype float16
    train_user_embeddings = train_user_embeddings.to(DTYPE)
    val_user_embeddings = val_user_embeddings.to(DTYPE)
    test_user_embeddings = test_user_embeddings.to(DTYPE)
    item_embeddings = item_embeddings.to(DTYPE)
    
    
    # negative sampling for training
    data_file = f"data/{args.data_name}/negative_samples.txt"
    neg_item_sets = []
    with open(data_file, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            user_id = int(parts[0])
            parts = [int(p) for p in parts[1:]]
            # randomly sample 10 negative items
            if len(parts) > args.num_neg_samples:
                parts = random.sample(parts, args.num_neg_samples)
            # parts = random.sample(parts, 30)
            neg_item_sets.append(parts)


    print("Merged train_user_embeddings.shape:", train_user_embeddings.shape)
    print("Merged val_user_embeddings.shape:", val_user_embeddings.shape)
    print("Merged test_user_embeddings.shape:", test_user_embeddings.shape)
    train_neg_item_sets = neg_item_sets

    train_anchor_ids = val_anchor_ids = test_anchor_ids = None
    if args.add_anchor_emb:
        print("Computing 60-day anchor items (most similar to user_emb; fallback: last item)...")
        history_60d_train = get_time_aware_user_history(args, "train")
        history_60d_val = get_time_aware_user_history(args, "valid")
        history_60d_test = get_time_aware_user_history(args, "test")
        train_anchor_ids = compute_anchor_item_ids(train_user_embeddings, train_user_seq_data, history_60d_train, item_embeddings, device)
        val_anchor_ids = compute_anchor_item_ids(val_user_embeddings, val_user_seq_data, history_60d_val, item_embeddings, device)
        test_anchor_ids = compute_anchor_item_ids(test_user_embeddings, test_user_seq_data, history_60d_test, item_embeddings, device)
        print(f"Anchor ids computed: train {len(train_anchor_ids)}, val {len(val_anchor_ids)}, test {len(test_anchor_ids)}")

    # Create DataLoader
    train_user_embeddings = train_user_embeddings[1:]
    val_user_embeddings = val_user_embeddings[1:]
    test_user_embeddings = test_user_embeddings[1:]

    def _train_anchor():
        if train_anchor_ids is None:
            return None
        if args.use_pre_train or args.use_pref:
            return train_anchor_ids + train_anchor_ids
        return train_anchor_ids

    if args.use_pre_train:
        pre_train_user_embeddings = pre_train_user_embeddings[1:]
        train_user_embeddings = torch.cat([train_user_embeddings, pre_train_user_embeddings], dim=0)
        train_pos_item_ids += pre_train_pos_item_ids
        new_neg_item_sets = neg_item_sets * 2
        train_loader = get_dataloader(train_user_embeddings, train_pos_item_ids, new_neg_item_sets, item_embeddings, batch_size=args.train_batch_size, filter_top_k=args.filter_top_k, data_ratio=args.train_data_ratio, filter_rank_csv=args.filter_rank_csv, filter_rank_max=args.filter_rank_max, anchor_item_ids=_train_anchor())
        del pre_train_user_embeddings, pre_train_pos_item_ids, new_neg_item_sets
        gc.collect()
    elif args.use_pref:
        train_pref_user_embeddings = train_pref_user_embeddings[1:]
        train_user_embeddings = torch.cat([train_user_embeddings, train_pref_user_embeddings], dim=0)
        train_pos_item_ids = train_pos_item_ids * 2
        new_neg_item_sets = neg_item_sets * 2
        train_loader = get_dataloader(train_user_embeddings, train_pos_item_ids, new_neg_item_sets, item_embeddings, batch_size=args.train_batch_size, filter_top_k=args.filter_top_k, data_ratio=args.train_data_ratio, filter_rank_csv=args.filter_rank_csv, filter_rank_max=args.filter_rank_max, anchor_item_ids=_train_anchor())
        del train_pref_user_embeddings, new_neg_item_sets
        gc.collect()
    else:
        train_loader = get_dataloader(
            train_user_embeddings, train_pos_item_ids, train_neg_item_sets, item_embeddings, batch_size=args.train_batch_size, filter_top_k=args.filter_top_k, data_ratio=args.train_data_ratio, filter_rank_csv=args.filter_rank_csv, filter_rank_max=args.filter_rank_max, anchor_item_ids=_train_anchor())
    gc.collect()
    valid_loader = get_dataloader(val_user_embeddings, val_pos_item_ids, neg_item_sets, item_embeddings, batch_size=args.eval_batch_size, shuffle=False, anchor_item_ids=val_anchor_ids)
    del val_user_embeddings, val_pos_item_ids
    gc.collect()
    test_loader = get_dataloader(test_user_embeddings, test_pos_item_ids, neg_item_sets, item_embeddings, batch_size=args.eval_batch_size, shuffle=False, anchor_item_ids=test_anchor_ids)
    del test_user_embeddings, test_pos_item_ids
    gc.collect()


    # Train the model
    if not args.zero_shot:
        trained_model = train_rec(
            args, model, train_loader, valid_loader, epochs=args.epochs, device=device
        )
    else:
        trained_model = model.set_zero_shot()

    # test the model
    test_loss, pred_user_embs, pred_item_embs = test_transformer_encoder(
        trained_model, test_loader, item_embeddings, device=device, use_anchor=args.add_anchor_emb
    )
    print("shape of user_pred_embs:", pred_user_embs.shape)
    print("shape of item_embs:", pred_item_embs.shape)

    # exit()

    # with open(f'sasrec_results/{args.crm_name}_{args.data_name}_test_user_id_2_top_{args.crm_top_k}_item_ids.json', 'r') as f:
    #     test_user_id_2_top_k_item_ids = json.load(f)
    # test_user_id_2_top_k_item_ids = {int(k): v for k, v in test_user_id_2_top_k_item_ids.items()}

    # Evaluation: evaluator와 동일한 방식 (batch retrieval metrics, cold/hot, novelty)
    pred_item_embs = pred_item_embs.to(device)
    pred_user_embs = pred_user_embs.to(device)
    if args.add_crm_emb:
        print("Using CRM embeddings")
        crm_item_embeddings, crm_train_user_embeddings, crm_val_user_embeddings, crm_test_user_embeddings = get_crm_embeddings(args)
        crm_item_embeddings = crm_item_embeddings.to(device)
        crm_test_user_embeddings = crm_test_user_embeddings.to(device)
        pred_item_embs = torch.cat([pred_item_embs, crm_item_embeddings], dim=-1)
        pred_user_embs = torch.cat([pred_user_embs, crm_test_user_embeddings[1:, :]], dim=-1)
        print("pred_item_embs.shape:", pred_item_embs.shape)
        print("pred_user_embs.shape:", pred_user_embs.shape)
    pred_item_embs = F.normalize(pred_item_embs, dim=-1)

    num_items = pred_item_embs.shape[0]
    item_popularity, item_novelty, cold_items, hot_items = load_item_popularity_novelty(args.data_name, num_items)
    # item_novelty is (num_items+1,) array; ensure index 0 is valid
    if len(item_novelty) <= num_items:
        item_novelty = np.concatenate([np.zeros(1), item_novelty])

    sorted_user_ids = sorted(test_user_seq_data.keys())
    targets = [test_user_target[uid] for uid in sorted_user_ids]
    histories = [test_user_seq_data[uid] for uid in sorted_user_ids]
    query_embeddings = pred_user_embs

    ks = [5, 10, 20]
    print("Computing retrieval metrics (evaluator-style)...")
    metrics, rank_info = compute_retrieval_metrics_embedding(
        query_embeddings,
        pred_item_embs,
        targets,
        histories,
        item_novelty,
        cold_items,
        hot_items,
        ks=ks,
        batch_size=args.eval_batch_size,
        device=device,
    )

    result_dict_list = []
    for j, user_id in enumerate(sorted_user_ids):
        ri = rank_info[j]
        history_iids = test_user_seq_data[user_id]
        top_indices = ri.get("top_500_indices", [])
        top_scores = ri.get("top_500_scores", [])
        result_dict_list.append({
            "user_id": user_id,
            "history_item_ids": " ".join([str(iid) for iid in history_iids]),
            "target_item_id": ri["target_item"],
            "is_cold_target": ri["is_cold"],
            "top_item_ids": " ".join([str(iid) for iid in top_indices[:500]]),
            "top_item_scores": " ".join([str(round(s, 2)) for s in top_scores[:500]]),
            "rank": ri["rank"],
        })

    avg_results = {k: float(round(np.mean(v), 4)) for k, v in metrics.items()}
    print("\n=== Overall Metrics ===")
    print("Average Metrics:", avg_results)

    cold_keys = [k for k in metrics if k.startswith("cold_")]
    if cold_keys and len(metrics.get("cold_hit@5", [])) > 0:
        avg_cold_results = {k: float(round(np.mean(metrics[k]), 4)) for k in cold_keys}
        print("\n=== Cold Item Metrics ===")
        print(f"Number of cold item targets: {len(metrics['cold_hit@5'])}")
        print("Cold Item Metrics:", avg_cold_results)
    else:
        avg_cold_results = {}
        print("\n=== Cold Item Metrics ===")
        print("No cold item targets found in test set")

    novelty_keys = [k for k in metrics if k.startswith("novelty")]
    avg_novelty_results = {k: float(round(np.mean(metrics[k]), 4)) for k in novelty_keys}
    print("\n=== Novelty Metrics ===")
    print("Novelty Metrics:", avg_novelty_results)

    output_file = f"results/embedding_training_{args.data_name}_{args.run_name}.log"
    all_metrics = {**avg_results, **avg_cold_results, **avg_novelty_results}
    max_key_length = max(len(key) for key in all_metrics.keys()) if all_metrics else 10

    with open(output_file, "a") as f:
        f.write("\n\nResults\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for key, value in vars(args).items():
            f.write(f"{key:<26} : {value}\n")
        f.write("---\n")
        f.write("=== Overall Metrics ===\n")
        for key, value in avg_results.items():
            f.write(f"{key.ljust(max_key_length)} : {value:.4f}\n")
        f.write("\n=== Cold Item Metrics ===\n")
        if avg_cold_results:
            f.write(f"Number of cold item targets: {len(metrics['cold_hit@5'])}\n")
            for key, value in avg_cold_results.items():
                f.write(f"{key.ljust(max_key_length)} : {value:.4f}\n")
        else:
            f.write("No cold item targets found in test set\n")
        f.write("\n=== Novelty Metrics ===\n")
        for key, value in avg_novelty_results.items():
            f.write(f"{key.ljust(max_key_length)} : {value:.4f}\n")
        f.write("\n=== Compact Summary ===\n")
        for key in ["hit@5", "ndcg@5", "hit@10", "ndcg@10"]:
            if key in avg_results:
                f.write(f"{avg_results[key]:.4f}\n")
        if avg_cold_results:
            f.write("\nCold Item Summary:\n")
            for key in ["cold_hit@5", "cold_ndcg@5", "cold_hit@10", "cold_ndcg@10"]:
                if key in avg_cold_results:
                    f.write(f"{avg_cold_results[key]:.4f}\n")
        f.write("\nNovelty Summary:\n")
        for key in ["novelty@5", "novelty@10"]:
            if key in avg_novelty_results:
                f.write(f"{avg_novelty_results[key]:.4f}\n")

    results_df = pd.DataFrame(result_dict_list)
    results_df.to_csv(f"results/embedding_training_{args.data_name}_{args.run_name}_{args.model_name}.csv", index=False)
    print(f"Metrics saved to {output_file}")

    # Master evaluation log (same format as zeroshot_trigger master log)
    master_log_path = f"results/0_{args.data_name}_projection.log"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(master_log_path, "a") as f:
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write(f"Master Evaluation Log - {args.run_name}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write("=" * 120 + "\n\n")
        f.write("[Overall Metrics]\n")
        f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
        f.write("-" * 70 + "\n")
        h5 = avg_results.get("hit@5", 0)
        n5 = avg_results.get("ndcg@5", 0)
        h10 = avg_results.get("hit@10", 0)
        n10 = avg_results.get("ndcg@10", 0)
        h20 = avg_results.get("hit@20", 0)
        n20 = avg_results.get("ndcg@20", 0)
        f.write(f"test\t{h5:.4f}\t{n5:.4f}\t{h10:.4f}\t{n10:.4f}\t{h20:.4f}\t{n20:.4f}\n\n")
        f.write("[Cold Items Metrics (80%)]\n")
        f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
        f.write("-" * 70 + "\n")
        if avg_cold_results:
            ch5 = avg_cold_results.get("cold_hit@5", 0)
            cn5 = avg_cold_results.get("cold_ndcg@5", 0)
            ch10 = avg_cold_results.get("cold_hit@10", 0)
            cn10 = avg_cold_results.get("cold_ndcg@10", 0)
            ch20 = avg_cold_results.get("cold_hit@20", 0)
            cn20 = avg_cold_results.get("cold_ndcg@20", 0)
            f.write(f"test\t{ch5:.4f}\t{cn5:.4f}\t{ch10:.4f}\t{cn10:.4f}\t{ch20:.4f}\t{cn20:.4f}\n\n")
        else:
            f.write("test\t-\t-\t-\t-\t-\t-\n\n")
        f.write("[Hot Items Metrics (Top 20%)]\n")
        f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
        f.write("-" * 70 + "\n")
        if avg_results.get("hot_hit@5") is not None:
            hh5 = avg_results.get("hot_hit@5", 0)
            hn5 = avg_results.get("hot_ndcg@5", 0)
            hh10 = avg_results.get("hot_hit@10", 0)
            hn10 = avg_results.get("hot_ndcg@10", 0)
            hh20 = avg_results.get("hot_hit@20", 0)
            hn20 = avg_results.get("hot_ndcg@20", 0)
            f.write(f"test\t{hh5:.4f}\t{hn5:.4f}\t{hh10:.4f}\t{hn10:.4f}\t{hh20:.4f}\t{hn20:.4f}\n\n")
        else:
            f.write("test\t-\t-\t-\t-\t-\t-\n\n")
        f.write("[Novelty Metrics]\n")
        f.write("Split\tNov@5\tNH@5\tNov@10\tNH@10\tNov@20\tNH@20\n")
        f.write("-" * 70 + "\n")
        nov5 = avg_novelty_results.get("novelty@5", 0)
        nh5 = avg_novelty_results.get("novelty_hit@5", 0)
        nov10 = avg_novelty_results.get("novelty@10", 0)
        nh10 = avg_novelty_results.get("novelty_hit@10", 0)
        nov20 = avg_novelty_results.get("novelty@20", 0)
        nh20 = avg_novelty_results.get("novelty_hit@20", 0)
        f.write(f"test\t{nov5:.4f}\t{nh5:.4f}\t{nov10:.4f}\t{nh10:.4f}\t{nov20:.4f}\t{nh20:.4f}\n")
        f.write("-" * 120 + "\n\n")
    print(f"Master log appended to {master_log_path}")