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
    """normalized title -> item_id reverse index"""
    index = {}
    for item_id, meta in item_metadata.items():
        title = meta.get("title", meta.get("name", ""))
        title = remove_html_tags(str(title)).strip()
        key = title.lower()
        if key and key not in index:
            index[key] = item_id
    return index


def parse_key_item(response: str) -> str:
    """Extract selected item name from <key_item> or <key_place> tag"""
    for tag in ("key_item", "key_place"):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def resolve_item_id(title_str: str, title_index: dict) -> tuple:
    """Generated title string → (item_id, title_str). Returns (None, title_str) if matching fails."""
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
    """CSV(user_id, selected_item_id) → {user_id: item_id} dictionary"""
    df = pd.read_csv(csv_path)
    mapping = {
        int(row["user_id"]): int(row["selected_item_id"])
        for _, row in df.iterrows()
        if pd.notna(row["selected_item_id"])
    }
    print(f"Loaded preselected items for {len(mapping)} users from {csv_path}")
    return mapping


def load_from_retrieval_rank_csv(csv_path: str, rank_threshold: int = 100) -> dict:
    """
    SASRec retrieval rank CSV (user_id, target_item_id, rank, best_rank_item_id, ...)
    selecting only users with rank <= rank_threshold,
    and returning best_rank_item_id as the postfix item.

    Returns: {user_id (int): best_rank_item_id (int)}
    """
    df = pd.read_csv(csv_path)
    required = {"user_id", "rank", "best_rank_item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from retrieval_rank_csv: {missing}")

    df = df.dropna(subset=["rank", "best_rank_item_id"])
    before = len(df)
    df = df[df["rank"] <= rank_threshold]
    print(
        f"Loaded retrieval rank CSV: {len(df)}/{before} users "
        f"with rank <= {rank_threshold} from {csv_path}"
    )
    return {int(row["user_id"]): int(row["best_rank_item_id"]) for _, row in df.iterrows()}

def filter_history_by_days(
    reviews: list, index: int, history_limit: int, days: int
) -> list:
    """
    Return only history items within `days` of the target item (reviews[index]).
    If no items fall within that range, return only the single most recent item.
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
    # fallback: single most recent item
    return candidates[-1:]

def load_best_from_ranked_csv(ranked_csv_path: str) -> dict:
    """
    Select the postfix_item_id with the lowest (= best) rank per user
    from a ranked CSV (user_id, postfix_item_id, rank, ...).
    Returns: {user_id (int): postfix_item_id (int)}
    """
    df = pd.read_csv(ranked_csv_path)
    required = {"user_id", "postfix_item_id", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from ranked_csv: {missing}")

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


def build_target_item_hint(
    item_metadata: dict,
    item_id: int,
    data_name: str,
) -> str:
    """
    Include the target item's title as a hint in the prompt.
    The generated query/preference text should not directly name the title,
    but only describe related keywords, features, and preferences.
    """
    title = get_item_title(item_metadata, item_id)

    is_yelp = (data_name == "yelp")
    subject     = "place" if is_yelp else "item"
    next_action = "visit" if is_yelp else "purchase"

    lines = [
        f"[Target {subject} for guidance — the user's next {next_action} is: {title}]",
        f"Using the above as a guiding signal: identify the single key {subject} "
        f"from the history that most strongly motivates purchasing \"{title}\", "
        f"then write a preference paragraph that naturally explains the user's underlying "
        f"preferences, related keywords, desired features, and specific needs — "
        f"without directly naming \"{title}\" in your output.",
    ]
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def process_key_selection(args):
    data_name      = args.data_name
    model_dir_name = args.model_name.split("/")[-1]

    # ranked_csv / retrieval_rank_csv / preselected_csv are mutually exclusive
    # Priority: retrieval_rank_csv > ranked_csv > preselected_csv
    use_suffix = (
        args.item_suffix
        or bool(args.preselected_csv)
        or bool(args.ranked_csv)
        or bool(args.retrieval_rank_csv)
    )
    use_target_guided = getattr(args, "target_guided", False)

    # Output filename
    suffix = ""
    if args.item_suffix:
        suffix += "_suffix"
    if args.retrieval_rank_csv:
        csv_stem = os.path.splitext(os.path.basename(args.retrieval_rank_csv))[0]
        suffix += f"_retrieval_rank_top{args.retrieval_rank_threshold}_from_{csv_stem}"
    elif args.ranked_csv:
        csv_stem = os.path.splitext(os.path.basename(args.ranked_csv))[0]
        suffix += f"_best_from_{csv_stem}"
    elif args.preselected_csv:
        csv_stem = os.path.splitext(os.path.basename(args.preselected_csv))[0]
        suffix += f"_from_{csv_stem}"
    if use_target_guided:
        suffix += "_target_guided"
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
    if args.retrieval_rank_csv:
        preselected_map = load_from_retrieval_rank_csv(
            args.retrieval_rank_csv, args.retrieval_rank_threshold
        )
    elif args.ranked_csv:
        preselected_map = load_best_from_ranked_csv(args.ranked_csv)
    elif args.preselected_csv:
        preselected_map = load_preselected_items(args.preselected_csv)

    # Tokenizer: apply chat template directly in postfix mode or target_guided mode
    need_tokenizer = use_suffix or use_target_guided
    tokenizer = None
    if need_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"Tokenizer loaded: {args.model_name}")

    # PromptGenerator (chat template is applied directly, so apply_chat_template=False)
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
        apply_chat_template=not need_tokenizer,  # apply directly in postfix/target_guided mode
    )

    split_index = {"train": -3, "valid": -2, "test": -1}
    index = split_index[args.split]

    prompts          = []
    user_ids         = []
    target_item_ids  = []
    postfix_item_ids = []   # None means no postfix

    for user_id, reviews in tqdm(user2reviews.items(), desc="Building prompts"):
        target_item_id = int(reviews[index]["item_id"])
        history_reviews = filter_history_by_days(
            reviews, index, args.history_limit, args.days
        )
        history_item_ids = [int(r["item_id"]) for r in history_reviews]
        assert len(history_item_ids) > 0, f"Empty history for user {user_id}"

        # Base history text
        history_text = prompt_generator.generate_prompt(
            item_ids=history_item_ids,
            user_id=user_id,
        )

        if use_target_guided:
            # target_guided mode: insert target item metadata as hint then apply chat template
            hint = build_target_item_hint(item_metadata, target_item_id, data_name)
            guided_user_text = history_text + "\n\n" + hint
            chat_str = apply_chat_template(tokenizer, guided_user_text)
            prompts.append(chat_str)
            user_ids.append(user_id)
            target_item_ids.append(target_item_id)
            postfix_item_ids.append(None)
        elif use_suffix:
            # Apply chat template → append item title postfix at end of string
            chat_str = apply_chat_template(tokenizer, history_text)

            if preselected_map:
                # When preselected_csv/retrieval_rank_csv is provided: 1 prompt per user
                presel_id = preselected_map.get(user_id)
                # Skip users not in the map (e.g., filtered out by rank threshold)
                if presel_id is None:
                    continue
                title = get_item_title(item_metadata, presel_id)
                prompts.append(chat_str + f"<key_item>{title}</key_item>")
                user_ids.append(user_id)
                target_item_ids.append(target_item_id)
                postfix_item_ids.append(presel_id)
            else:
                # --item_suffix: N history items → N prompts
                for item_id in history_item_ids:
                    title = get_item_title(item_metadata, item_id)
                    prompts.append(chat_str + f"<key_item>{title}</key_item>")
                    user_ids.append(user_id)
                    target_item_ids.append(target_item_id)
                    postfix_item_ids.append(item_id)
        else:
            # Default mode: LLMGenerator applies chat template
            prompts.append(history_text)
            user_ids.append(user_id)
            target_item_ids.append(target_item_id)
            postfix_item_ids.append(None)

    # Print prompt samples
    sample_idx = [i for i in [0, min(10, len(prompts)-1), min(20, len(prompts)-1)]
                  if i < len(prompts)]
    print("\n--- Prompt samples ---")
    for i in sample_idx:
        print(f"[index {i}]\n{prompts[i]}\n{'-'*60}")

    # postfix mode: compare base (chat template only) vs final (+postfix)
    if use_suffix and prompts:
        print("\n--- item_suffix example ---")
        i = sample_idx[0]
        uid = user_ids[i]
        h_ids = [int(r["item_id"])
                 for r in user2reviews[uid][index - args.history_limit : index]]
        base_text = prompt_generator.generate_prompt(item_ids=h_ids, user_id=uid)
        chat_base = apply_chat_template(tokenizer, base_text)
        postfix_title = get_item_title(item_metadata, postfix_item_ids[i]) \
                        if postfix_item_ids[i] else "(none)"
        print(f"[After chat template (base)]\n{chat_base}")
        print(f"\n[postfix title] '{postfix_title}'")
        print(f"\n[final prompt]\n{prompts[i]}")
        print("=" * 60)

    # target_guided mode: print example of hint insertion followed by chat template
    if use_target_guided and prompts:
        print("\n--- target_guided example ---")
        i = sample_idx[0]
        uid = user_ids[i]
        tid = target_item_ids[i]
        h_ids = [int(r["item_id"])
                 for r in user2reviews[uid][index - args.history_limit : index]]
        hint = build_target_item_hint(item_metadata, tid, data_name)
        print(f"[Target item title] '{get_item_title(item_metadata, tid)}'")
        print(f"\n[Inserted hint]\n{hint}")
        print(f"\n[final prompt]\n{prompts[i]}")
        print("=" * 60)

    # Generate
    n_samples = getattr(args, 'n_samples', 1)
    llm_generator = LLMGenerator(args)
    responses, token_nums = llm_generator.generate_response(
        KEY_SELECTION_SYSTEM_PROMPT, prompts,
        raw_prompt=(use_suffix or use_target_guided),
        n_samples=n_samples,
    )

    # When n_samples==1 it is a flat list of strings; when >1, a list of lists.
    # Normalize to always list[list[str]]
    if n_samples == 1:
        responses_nested = [[r] for r in responses]
    else:
        responses_nested = responses

    print("\n--- Response samples ---")
    for i in sample_idx:
        for s_idx, r in enumerate(responses_nested[i]):
            print(f"[index {i}, sample {s_idx}]\n{r}\n{'-'*60}")

    # Save results (N rows per user when n_samples > 1)
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
        help="Use prompts as-is in LLMGenerator without applying the chat template."
    )
    parser.add_argument(
        "--n_samples", type=int, default=1,
        help="Number of samples to generate per prompt (vLLM SamplingParams n). Results saved to a single CSV."
    )
    parser.add_argument(
        "--item_suffix", action="store_true",
        help=(
            "After applying the chat template, append history item titles as postfix. "
            "N history items → N prompts per user."
        ),
    )
    parser.add_argument(
        "--preselected_csv", type=str, default=None,
        help=(
            "Path to a CSV with user_id and selected_item_id columns. "
            "The item title is used as postfix after the chat template (1 prompt per user)."
        ),
    )
    parser.add_argument(
        "--ranked_csv", type=str, default=None,
        help=(
            "Path to the ranked CSV generated by eval_key_selection.py "
            "(requires user_id, postfix_item_id, rank columns). "
            "Automatically selects the postfix_item_id with the lowest rank per user as suffix."
        ),
    )
    parser.add_argument(
        "--retrieval_rank_csv", type=str, default=None,
        help=(
            "Path to the CSV generated by find_trigger_items_from_sequential.py (mode=rank) "
            "(requires user_id, target_item_id, rank, best_rank_item_id columns). "
            "Only users with rank <= --retrieval_rank_threshold are selected, "
            "and best_rank_item_id is used as postfix."
        ),
    )
    parser.add_argument(
        "--retrieval_rank_threshold", type=int, default=100,
        help="Maximum rank value to include when using --retrieval_rank_csv (default: 100).",
    )
    parser.add_argument(
        "--target_guided", action="store_true",
        help=(
            "Insert the target (ground-truth) item's category/brand/description as a hint into the user prompt. "
            "The title is excluded so the correct answer is not directly revealed. "
            "The model generates a key item and user preference text that can recommend the target item."
        ),
    )

    args = parser.parse_args()

    import torch
    print(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    process_key_selection(args)
