import json
import html
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from transformers import AutoTokenizer
from llm_generator import LLMGenerator

from utils.dataset import PromptGenerator

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


USER_PREFERENCE_SYSTEM_PROMPT = """
You are an intelligent assistant that analyzes a user's recent preferences based on their chronological purchase history.
""".strip()


import re

def remove_html_tags(text):
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def process_item_features(args):
    """
    Processes item features to extract title embeddings for a specified dataset,
    and optionally generates new item summaries using an LLM.
    """
    # Load data and configure paths
    data_name = args.data_name
    model_name = args.model_name

    model_dir_name = model_name.split("/")[-1]
    input_file = f"data/{data_name}/meta_text_fix.json"
    output_file = f"data_processed/{data_name}_{model_dir_name}_{args.split}_{args.prompt_type}.json"

    print(output_file)
    # Load item features
    with open(input_file, 'r') as f:
        item_metadata = json.load(f)
    item_metadata = {int(k): v for k, v in item_metadata.items()}

    reviews_file = f"data/{data_name}/user2reviews_with_date.json"
    with open(reviews_file, 'r') as f:
        user2reviews = json.load(f)
    user2reviews = {int(k): v for k, v in user2reviews.items()}

    # Create PromptGenerator
    prompt_generator = PromptGenerator(
        item_metadata=item_metadata,
        data_name=data_name,
        prompt_type=args.prompt_type,
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
    )

    # Generate prompts
    uid_2_history_text = {}
    if args.split == "train":
        index = -3
    elif args.split == "valid":
        index = -2
    elif args.split == "test":
        index = -1
    
    for user_id, reviews in user2reviews.items():
        target_timestamp = int(reviews[index]["timestamp"])
        # Extract history item IDs
        history_item_ids = [int(review["item_id"]) for review in reviews[index-args.history_limit:index]]

        assert len(history_item_ids) > 0, f"History item ids is empty for user {user_id}"
        
        # Generate history text using PromptGenerator
        history_text = prompt_generator.generate_prompt(
            item_ids=history_item_ids,
            user_id=user_id,
            target_timestamp=target_timestamp
        )
        
        uid_2_history_text[user_id] = history_text

    # Generate LLM-based summaries
    print("Generating item summaries using LLM...")
    prompts = []
    user_ids = []
    for user_id, user_history_text in tqdm(uid_2_history_text.items()):
        # Prepare prompt with item features
        prompt = user_history_text
        prompts.append(prompt)
        user_ids.append(user_id)

    print("prompt sample:")
    for i in [10, 20, 30]:
        print(prompts[i])

    # prompts = prompts[:10]

    llm_generator = LLMGenerator(args)
    responses, token_nums = llm_generator.generate_response(USER_PREFERENCE_SYSTEM_PROMPT, prompts)

    print("Response sample:")
    for i in [10, 20, 30]:
        print(f"Response {i}:")
        print(responses[i])
        print("------\n")
    # # Generate responses using vLLM
    # model = get_vllm_model(args.model_name, args.num_gpus)
    # params = get_sampling_params(args)
    # responses = generate_response_with_vllm(model, prompts, params)


    # Update item features with generated summaries
    user_summary = {}
    for item_id, summary in zip(user_ids, responses):
        user_summary[item_id] = summary.strip()
    
    # Save item summaries to file
    with open(output_file, 'w') as f:
        json.dump(user_summary, f, indent=4)

if __name__ == "__main__":
    random.seed(42)
    import argparse
    parser = argparse.ArgumentParser(description="Run inference and generate test results.")
    parser.add_argument("--run_name", type=str, default="basic", help="Name of the dataset to use.")
    parser.add_argument("--data_name", type=str, default="beauty", help="Name of the dataset to use.")
    parser.add_argument("--check_interval", type=int, default=100, help="Number of samples to use for testing.")
    
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the model to use.")    
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for sampling.")
    parser.add_argument("--max_input_tokens", type=int, default=1024*6, help="Max tokens for sampling.")
    parser.add_argument("--max_output_tokens", type=int, default=256, help="Max tokens for sampling.")
    parser.add_argument("--word_limit", type=int, default=128, help="")
    parser.add_argument("--history_limit", type=int, default=8, help="")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for sampling.")
    parser.add_argument("--prompt_type", type=str, default="seq_rec", help="Type of prompt to use.")
    parser.add_argument("--days", type=int, default=365, help="Days to use for user preference generation.")

    parser.add_argument("--use_reviews", action="store_true", help="Use reviews for user preference generation.")
    parser.add_argument("--max_words", type=int, default=128, help="Max words for user preference generation.")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing item features.")
    parser.add_argument("--use_deep_infra", action="store_true", help="Use Deep Infra for inference.")
    parser.add_argument("--split", type=str, default="test", help="Split to use (train, valid, test).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="GPU memory utilization for inference.")

    args = parser.parse_args()

    import torch

    # allocate 1gb tensor on cuda
    # tensor = torch.zeros(1024**3, dtype=torch.float32, device="cuda")
    # print(f"Tensor size: {tensor.shape}")

    # print cuda memory usage
    print(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory/1024**3} GB")


    process_item_features(args)