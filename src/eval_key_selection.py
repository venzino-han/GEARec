#!/usr/bin/env python3
"""
eval_key_selection.py

generate_key_selection.py로 생성된 CSV를 입력으로 받아,
각 행의 "선택된 키 아이템 타이틀 + <query> 태그 텍스트" 임베딩을 계산하고,
정답 아이템(target_item_id)의 RANK를 측정하여 컬럼으로 추가한 뒤 저장.
"""

import os
import gc
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluator import RecommendationEvaluator
from utils.reward_function import extract_query_from_tags


# ─── 텍스트 결합 ──────────────────────────────────────────────────────────────

def build_embedding_text(row: pd.Series, key_col: str) -> str:
    """
    키 아이템 타이틀 + <query> 태그 텍스트를 결합해 임베딩용 문자열 반환.
    key_col: 'selected_item_title' 또는 'postfix_item_title'
    """
    key_item  = str(row.get(key_col, "") or "").strip()
    generated = str(row.get("generated_text", "") or "").strip()
    query     = extract_query_from_tags(generated, tag="query")

    if key_item:
        return f"{key_item}\n{query}"
    return query


# ─── 히스토리 로드 ────────────────────────────────────────────────────────────

def load_histories(data_name: str, user_ids: list, split: str, history_limit: int) -> list:
    """user2reviews_with_date.json 에서 각 유저의 히스토리 아이템 ID 리스트를 반환."""
    with open(f"data/{data_name}/user2reviews_with_date.json", "r") as f:
        user2reviews = {int(k): v for k, v in json.load(f).items()}

    split_index = {"train": -3, "valid": -2, "test": -1}
    idx = split_index[split]

    histories = []
    for uid in user_ids:
        reviews = user2reviews.get(int(uid), [])
        h_ids = [int(r["item_id"]) for r in reviews[idx - history_limit : idx]]
        histories.append(h_ids)
    return histories


# ─── Rank 계산 (배치) ─────────────────────────────────────────────────────────

def compute_ranks(
    query_embeddings: torch.Tensor,
    targets: list,
    histories: list,
    item_embeddings: torch.Tensor,
    batch_size: int = 512,
) -> list:
    """
    query_embeddings : [N, D]
    targets          : list[int]  target_item_id (1-indexed)
    histories        : list[list[int]]
    item_embeddings  : [num_items, D]  normalized

    Returns: list of dict with keys {rank, target_score}
    """
    device     = query_embeddings.device
    n_queries  = query_embeddings.shape[0]
    rank_info  = []

    # query 도 정규화
    norms = query_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    query_embeddings = query_embeddings / norms

    for start in tqdm(range(0, n_queries, batch_size), desc="Computing ranks"):
        end  = min(start + batch_size, n_queries)
        q    = query_embeddings[start:end]                      # [B, D]
        scores = torch.matmul(q, item_embeddings.T)             # [B, num_items]

        for i in range(end - start):
            global_i = start + i
            target   = targets[global_i]
            hist     = histories[global_i]

            # 히스토리 아이템(타겟 제외) 점수를 -inf 로
            excl = [h for h in hist if h != target]
            if excl:
                scores[i, excl] = float("-inf")

            target_score = scores[i, target].item()
            rank = int((scores[i] > target_score).sum().item()) + 1

            rank_info.append({"rank": rank, "target_score": target_score})

    return rank_info


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main(args):
    print(f"Input CSV : {args.input_csv}")

    # CSV 로드
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")

    # 키 아이템 컬럼 결정
    #   postfix_item_title 이 있으면 우선 사용, 없으면 selected_item_title
    if "postfix_item_title" in df.columns and df["postfix_item_title"].notna().any():
        key_col = "postfix_item_title"
    else:
        key_col = "selected_item_title"
    print(f"Key-item column : {key_col}")

    # 임베딩용 텍스트 생성
    embedding_texts = [build_embedding_text(row, key_col) for _, row in df.iterrows()]
    print("Sample embedding text:")
    print(embedding_texts[0][:300])

    # RecommendationEvaluator 생성 (임베딩 모델 + 아이템 임베딩 로드)
    evaluator = RecommendationEvaluator(args, checkpoint_dir=None)

    # 쿼리 임베딩 계산
    #   compute_embeddings 내부에서 <query> 태그를 다시 추출하지 않도록
    #   이미 결합된 텍스트를 직접 사용하기 위해 raw 버전으로 호출
    evaluator._load_embedding_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing embeddings for {len(embedding_texts)} texts...")

    if evaluator.use_sentence_transformers:
        raw_embs = evaluator.emb_llm.encode(
            embedding_texts,
            batch_size=args.eval_emb_batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        query_embeddings = raw_embs.to(device)
    else:
        from vllm.pooling_params import PoolingParams
        pooling_params = PoolingParams(
            truncate_prompt_tokens=args.eval_emb_max_length,
            task="embed",
        )
        texts_with_cls = [f"[CLS] {t}" for t in embedding_texts]
        emb_outputs = evaluator.emb_llm.encode(
            prompts=texts_with_cls,
            pooling_task="embed",
            pooling_params=pooling_params,
            use_tqdm=True,
        )
        query_embeddings = torch.stack([
            torch.as_tensor(o.outputs.data, dtype=torch.float32, device=device)
            for o in emb_outputs
        ])

    # 임베딩 모델 해제
    del evaluator.emb_llm
    evaluator.emb_llm = None
    torch.cuda.empty_cache()
    gc.collect()

    # 히스토리 로드
    user_ids = df["user_id"].tolist()
    targets  = df["target_item_id"].astype(int).tolist()
    histories = load_histories(args.data_name, user_ids, args.split, args.history_limit)

    # 아이템 임베딩을 같은 device로
    item_embs = evaluator.item_embeddings.to(device)

    # Rank 계산
    rank_info = compute_ranks(
        query_embeddings, targets, histories, item_embs,
        batch_size=args.eval_batch_size,
    )

    # 결과 컬럼 추가
    df["rank"]         = [r["rank"]         for r in rank_info]
    df["target_score"] = [r["target_score"] for r in rank_info]
    df["hit@5"]        = (df["rank"] <= 5).astype(int)
    df["hit@10"]       = (df["rank"] <= 10).astype(int)
    df["hit@20"]       = (df["rank"] <= 20).astype(int)
    df["embedding_text"] = embedding_texts

    # 출력 파일
    input_path  = Path(args.input_csv)
    output_path = input_path.with_name(input_path.stem + "_ranked.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    # 통계 출력
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Total rows     : {len(df)}")
    print(f"  Mean rank      : {df['rank'].mean():.1f}")
    print(f"  Median rank    : {df['rank'].median():.0f}")
    print(f"  Hit@5          : {df['hit@5'].mean():.4f}")
    print(f"  Hit@10         : {df['hit@10'].mean():.4f}")
    print(f"  Hit@20         : {df['hit@20'].mean():.4f}")
    print(f"{'='*60}")

    return df


# ─── argparse ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate key-selection CSV by computing target-item rank."
    )

    # 필수 인자
    parser.add_argument("--input_csv", type=str, required=True,
                        help="generate_key_selection.py 가 생성한 CSV 경로")
    parser.add_argument("--data_name", type=str, required=True,
                        help="데이터셋 이름 (beauty, sports, toys, yelp)")

    # 데이터
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--history_limit", type=int, default=8)

    # 임베딩 모델
    parser.add_argument("--emb_model_name", type=str,
                        default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--emb_type", type=str,
                        default="item_preference_1024_gemma-3-4b-it",
                        help="아이템 임베딩 파일 타입 (data_emb/{data}_{emb_type}_{model}.pt)")
    parser.add_argument("--use_sentence_transformers", action="store_true", default=True)
    parser.add_argument("--eval_emb_max_length", type=int, default=512)
    parser.add_argument("--eval_emb_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size",     type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

    # 기타
    parser.add_argument("--run_name", type=str, default="key_selection_eval")

    args = parser.parse_args()

    print(f"CUDA total memory: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    main(args)
