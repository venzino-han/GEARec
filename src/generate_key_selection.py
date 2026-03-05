import json
import os
import re
import random
import argparse

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from llm_generator import LLMGenerator
from utils.dataset import PromptGenerator


KEY_SELECTION_SYSTEM_PROMPT = """
You are an intelligent shopping assistant that analyzes a user's purchase history to identify key purchase drivers.
""".strip()


def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)


def get_item_title(item_metadata: dict, item_id: int) -> str:
    meta = item_metadata.get(item_id, {})
    title = meta.get("title", meta.get("name", ""))
    return remove_html_tags(str(title)).strip()


def build_title_index(item_metadata: dict) -> dict:
    """normalized title -> item_id 역인덱스"""
    index = {}
    for item_id, meta in item_metadata.items():
        title = meta.get("title", meta.get("name", ""))
        title = remove_html_tags(str(title)).strip()
        key = title.lower()
        if key and key not in index:
            index[key] = item_id
    return index


def parse_key_item(response: str) -> str:
    """<key_item> 또는 <key_place> 태그에서 선택된 아이템 이름 추출"""
    for tag in ("key_item", "key_place"):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def resolve_item_id(title_str: str, title_index: dict) -> tuple:
    """생성된 타이틀 문자열 → (item_id, title_str). 매칭 실패 시 (None, title_str)"""
    key = title_str.lower().strip()
    if not key:
        return None, title_str
    if key in title_index:
        return title_index[key], title_str
    for idx_key, item_id in title_index.items():
        if key in idx_key or idx_key in key:
            return item_id, title_str
    return None, title_str


def load_preselected_items(csv_path: str) -> dict:
    """CSV(user_id, selected_item_id) → {user_id: item_id} 딕셔너리"""
    df = pd.read_csv(csv_path)
    mapping = {
        int(row["user_id"]): int(row["selected_item_id"])
        for _, row in df.iterrows()
        if pd.notna(row["selected_item_id"])
    }
    print(f"Loaded preselected items for {len(mapping)} users from {csv_path}")
    return mapping


def filter_history_by_days(
    reviews: list, index: int, history_limit: int, days: int
) -> list:
    """
    target 아이템(reviews[index]) 기준으로 days 이내의 히스토리 아이템만 반환.
    해당 범위 내 아이템이 없으면 가장 최근 아이템 1개만 반환.
    """
    target_ts = int(reviews[index].get("timestamp", 0))
    candidates = reviews[index - history_limit : index]

    if target_ts <= 0 or days <= 0:
        return candidates

    threshold = days * 24 * 60 * 60
    filtered = [
        r for r in candidates
        if target_ts - int(r.get("timestamp", 0)) <= threshold
    ]
    if filtered:
        return filtered
    # fallback: 가장 최근 아이템 1개
    return candidates[-1:]


def load_best_from_ranked_csv(ranked_csv_path: str) -> dict:
    """
    ranked CSV(user_id, postfix_item_id, rank, ...) 에서
    유저별 rank 가 가장 낮은(= 가장 좋은) postfix_item_id 를 선정.
    Returns: {user_id (int): postfix_item_id (int)}
    """
    df = pd.read_csv(ranked_csv_path)
    required = {"user_id", "postfix_item_id", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ranked_csv에 필요한 컬럼이 없습니다: {missing}")

    df = df.dropna(subset=["postfix_item_id", "rank"])
    best = (
        df.sort_values("rank")
          .groupby("user_id", as_index=False)
          .first()[["user_id", "postfix_item_id"]]
    )
    mapping = {
        int(row["user_id"]): int(row["postfix_item_id"])
        for _, row in best.iterrows()
    }
    print(f"Extracted best postfix for {len(mapping)} users from {ranked_csv_path}")
    return mapping


def apply_chat_template(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": KEY_SELECTION_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── 메인 ────────────────────────────────────────────────────────────────────

def process_key_selection(args):
    data_name      = args.data_name
    model_dir_name = args.model_name.split("/")[-1]

    # ranked_csv 와 preselected_csv 는 상호 배타적으로 동작
    # ranked_csv 가 우선
    use_suffix = args.item_suffix or bool(args.preselected_csv) or bool(args.ranked_csv)

    # 출력 파일명
    suffix = ""
    if args.item_suffix:
        suffix += "_suffix"
    if args.ranked_csv:
        csv_stem = os.path.splitext(os.path.basename(args.ranked_csv))[0]
        suffix += f"_best_from_{csv_stem}"
    elif args.preselected_csv:
        csv_stem = os.path.splitext(os.path.basename(args.preselected_csv))[0]
        suffix += f"_from_{csv_stem}"
    output_file = (
        f"data_processed/{data_name}_{model_dir_name}_{args.split}"
        f"_key_selection{suffix}.csv"
    )
    print(f"Output: {output_file}")

    # Load item metadata
    with open(f"data/{data_name}/meta_text_fix.json", "r") as f:
        item_metadata = {int(k): v for k, v in json.load(f).items()}

    with open(f"data/{data_name}/user2reviews_with_date.json", "r") as f:
        user2reviews = {int(k): v for k, v in json.load(f).items()}

    title_index = build_title_index(item_metadata)

    preselected_map: dict = {}
    if args.ranked_csv:
        # ranked CSV에서 유저별 best postfix 자동 추출
        preselected_map = load_best_from_ranked_csv(args.ranked_csv)
    elif args.preselected_csv:
        preselected_map = load_preselected_items(args.preselected_csv)

    # 토크나이저 (postfix 모드에서 챗 템플릿 직접 적용)
    tokenizer = None
    if use_suffix:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"Tokenizer loaded: {args.model_name}")

    # PromptGenerator (챗 템플릿은 직접 적용하므로 apply_chat_template=False)
    prompt_generator = PromptGenerator(
        item_metadata=item_metadata,
        data_name=data_name,
        prompt_type="key_selection",
        use_brand=True,
        use_category=True,
        use_description=False,
        use_features=False,
        use_last_item=False,
        use_date=True,
        max_history_len=args.history_limit,
        history_text_max_length=args.word_limit,
        use_reviews=args.use_reviews,
        days_filter=args.days,
        apply_chat_template=not use_suffix,  # postfix 모드에서는 직접 적용
    )

    split_index = {"train": -3, "valid": -2, "test": -1}
    index = split_index[args.split]

    prompts          = []
    user_ids         = []
    target_item_ids  = []
    postfix_item_ids = []   # None이면 postfix 없음

    for user_id, reviews in tqdm(user2reviews.items(), desc="Building prompts"):
        target_item_id = int(reviews[index]["item_id"])
        history_reviews = filter_history_by_days(
            reviews, index, args.history_limit, args.days
        )
        history_item_ids = [int(r["item_id"]) for r in history_reviews]
        assert len(history_item_ids) > 0, f"Empty history for user {user_id}"

        # 기본 히스토리 텍스트
        history_text = prompt_generator.generate_prompt(
            item_ids=history_item_ids,
            user_id=user_id,
        )

        if use_suffix:
            # 챗 템플릿 적용 → 문자열 끝에 아이템 타이틀 postfix
            chat_str = apply_chat_template(tokenizer, history_text)

            if preselected_map:
                # preselected_csv 제공 시: 유저당 1개 프롬프트
                presel_id = preselected_map.get(user_id)
                title = get_item_title(item_metadata, presel_id) if presel_id else ""
                prompts.append(chat_str + f"<key_item>{title}</key_item>")
                user_ids.append(user_id)
                target_item_ids.append(target_item_id)
                postfix_item_ids.append(presel_id)
            else:
                # --item_suffix: 히스토리 아이템 N개 → N개 프롬프트
                for item_id in history_item_ids:
                    title = get_item_title(item_metadata, item_id)
                    prompts.append(chat_str + f"<key_item>{title}</key_item>")
                    user_ids.append(user_id)
                    target_item_ids.append(target_item_id)
                    postfix_item_ids.append(item_id)
        else:
            # 기본 모드: LLMGenerator가 챗 템플릿 적용
            prompts.append(history_text)
            user_ids.append(user_id)
            target_item_ids.append(target_item_id)
            postfix_item_ids.append(None)

    # 프롬프트 샘플 출력
    sample_idx = [i for i in [0, min(10, len(prompts)-1), min(20, len(prompts)-1)]
                  if i < len(prompts)]
    print("\n--- Prompt samples ---")
    for i in sample_idx:
        print(f"[index {i}]\n{prompts[i]}\n{'-'*60}")

    # postfix 모드: base(챗 템플릿만) vs 최종(+postfix) 비교 출력
    if use_suffix and prompts:
        print("\n--- item_suffix 적용 예시 ---")
        i = sample_idx[0]
        uid = user_ids[i]
        h_ids = [int(r["item_id"])
                 for r in user2reviews[uid][index - args.history_limit : index]]
        base_text = prompt_generator.generate_prompt(item_ids=h_ids, user_id=uid)
        chat_base = apply_chat_template(tokenizer, base_text)
        postfix_title = get_item_title(item_metadata, postfix_item_ids[i]) \
                        if postfix_item_ids[i] else "(none)"
        print(f"[챗 템플릿 적용 후 (base)]\n{chat_base}")
        print(f"\n[postfix 타이틀] '{postfix_title}'")
        print(f"\n[최종 prompt]\n{prompts[i]}")
        print("=" * 60)

    # 생성
    n_samples = getattr(args, 'n_samples', 1)
    llm_generator = LLMGenerator(args)
    responses, token_nums = llm_generator.generate_response(
        KEY_SELECTION_SYSTEM_PROMPT, prompts,
        raw_prompt=use_suffix,
        n_samples=n_samples,
    )

    # n_samples==1 이면 단일 문자열 리스트, >1 이면 리스트의 리스트
    # 통일을 위해 항상 list[list[str]] 형태로 변환
    if n_samples == 1:
        responses_nested = [[r] for r in responses]
    else:
        responses_nested = responses

    print("\n--- Response samples ---")
    for i in sample_idx:
        for s_idx, r in enumerate(responses_nested[i]):
            print(f"[index {i}, sample {s_idx}]\n{r}\n{'-'*60}")

    # 결과 저장 (n_samples > 1이면 유저당 N행)
    rows = []
    for prompt, user_id, target_item_id, postfix_id, sample_responses in zip(
        prompts, user_ids, target_item_ids, postfix_item_ids, responses_nested
    ):
        for s_idx, response in enumerate(sample_responses):
            generated_text = response.strip()
            selected_title_raw = parse_key_item(generated_text)
            selected_item_id, selected_item_title = resolve_item_id(
                selected_title_raw, title_index
            )
            rows.append({
                "user_id":             user_id,
                "target_item_id":      target_item_id,
                "target_item_title":   get_item_title(item_metadata, target_item_id),
                "postfix_item_id":     postfix_id,
                "postfix_item_title":  get_item_title(item_metadata, postfix_id) if postfix_id else "",
                "selected_item_id":    selected_item_id,
                "selected_item_title": selected_item_title,
                "sample_idx":          s_idx,
                "prompt":              prompt,
                "generated_text":      generated_text,
            })

    df = pd.DataFrame(rows, columns=[
        "user_id", "target_item_id", "target_item_title",
        "postfix_item_id", "postfix_item_title",
        "selected_item_id", "selected_item_title",
        "sample_idx",
        "prompt", "generated_text",
    ])
    df.to_csv(output_file, index=False)

    print(f"\nSaved {len(df)} rows to {output_file}")
    matched = df["selected_item_id"].notna().sum()
    print(f"Key-item matched: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")


# ─── argparse ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser(
        description="Generate key-item selection queries from user purchase history."
    )
    parser.add_argument("--run_name",   type=str,   default="key_selection")
    parser.add_argument("--data_name",  type=str,   default="beauty")
    parser.add_argument("--model_name", type=str,   default="google/gemma-3-1b-it")
    parser.add_argument("--temperature",            type=float, default=0.01)
    parser.add_argument("--max_input_tokens",       type=int,   default=1024 * 6)
    parser.add_argument("--max_output_tokens",      type=int,   default=256)
    parser.add_argument("--word_limit",             type=int,   default=128)
    parser.add_argument("--history_limit",          type=int,   default=8)
    parser.add_argument("--dtype",      type=str,   default="bfloat16")
    parser.add_argument("--days",       type=int,   default=365)
    parser.add_argument("--use_reviews",            action="store_true")
    parser.add_argument("--batch_size",             type=int,   default=8)
    parser.add_argument("--use_deep_infra",         action="store_true")
    parser.add_argument("--split",  type=str,   default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--num_gpus",               type=int,   default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--check_interval",         type=int,   default=100)

    parser.add_argument(
        "--raw_prompt", action="store_true",
        help="LLMGenerator에서 챗 템플릿을 적용하지 않고 프롬프트를 그대로 사용."
    )
    parser.add_argument(
        "--n_samples", type=int, default=1,
        help="프롬프트당 생성할 샘플 수 (vLLM SamplingParams n). 결과는 단일 CSV로 저장."
    )
    parser.add_argument(
        "--item_suffix", action="store_true",
        help=(
            "챗 템플릿 적용 후 히스토리 아이템 타이틀을 postfix로 붙여 생성. "
            "히스토리 아이템 N개 → 유저당 N개 프롬프트."
        ),
    )
    parser.add_argument(
        "--preselected_csv", type=str, default=None,
        help=(
            "user_id, selected_item_id 컬럼을 가진 CSV 경로. "
            "해당 아이템 타이틀을 챗 템플릿 이후 postfix로 사용 (유저당 1개 프롬프트)."
        ),
    )
    parser.add_argument(
        "--ranked_csv", type=str, default=None,
        help=(
            "eval_key_selection.py 가 생성한 ranked CSV 경로 "
            "(user_id, postfix_item_id, rank 컬럼 필요). "
            "유저별 rank 최소 postfix_item_id 를 자동 선정하여 suffix로 사용."
        ),
    )

    args = parser.parse_args()

    import torch
    print(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    process_key_selection(args)
