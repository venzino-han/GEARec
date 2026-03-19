"""
Dataset utilities for RL4Rec
Dataset and DataLoader utilities for recommendation system training
"""

import os
import json
import pickle
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from utils.prompt_templates import PROMPT_TEMPLATES, PROMPT_TEMPLATES_YELP

class PromptGenerator:
    """
    Class for generating prompts from user sequences.

    Supported prompt types:
    - 'preference': describe user preferences
    - 'next_item': predict next item
    - 'recommendation': generate recommended items
    - 'user_profile': generate user profile
    - 'recent_preference': describe recent preferences
    """
    
    def __init__(
        self,
        item_metadata: Dict,
        data_name: str = None,
        prompt_type: str = 'seq_rec',
        use_brand: bool = True,
        use_category: bool = True,
        use_description: bool = False,
        use_features: bool = False,
        use_last_item: bool = False,
        use_date: bool = True,
        max_history_len: int = 8,
        history_text_max_length: int = 100,
        use_reviews: bool = False,
        days_filter: int = None,
        tokenizer = None,
        apply_chat_template: bool = True,
        emphasize_recent_item: bool = False,
        include_target_date: bool = False,
        use_sasrec: bool = False,
        sasrec_top_k: int = 5,
        use_relative_date: bool = False,
        trigger_items: Optional[Dict[int, int]] = None,
        trigger_emphasis_text: Optional[str] = None,
    ):
        """
        Args:
            item_metadata: item metadata dictionary
            data_name: dataset name (used for loading date information)
            prompt_type: prompt type ('preference', 'next_item', 'recommendation', 'user_profile', 'recent_preference', 'reasoning')
            use_brand: whether to include brand information
            use_category: whether to include category information
            use_description: whether to include description information
            use_features: whether to include feature information
            use_last_item: whether to emphasize the last item
            use_date: whether to include date information (used for history and recent purchase emphasis)
            max_history_len: maximum history length
            history_text_max_length: maximum number of words in history text (also applied to review text)
            use_reviews: whether to include review text
            days_filter: include only reviews within the last N days (None means no filtering)
            tokenizer: tokenizer (required for applying chat template, optional)
            apply_chat_template: whether to apply chat template
            emphasize_recent_item: whether to emphasize the most recently purchased item in detail ("This user's most recent purchase is..." format; includes purchase date when use_date is True)
            include_target_date: whether to include the purchase date of the target/label item at the end of the prompt
            use_sasrec: whether to include SASRec recommendation results in the prompt
            sasrec_top_k: include only the top-K items from SASRec recommendation results
            use_relative_date: whether to use relative date notation (when True, displays as "(D-10)", "(D-20)" relative to the target date)
        """
        self.item_metadata = item_metadata
        self.data_name = data_name
        self.use_brand = use_brand
        self.use_category = use_category
        self.use_description = use_description
        self.use_features = use_features
        self.use_last_item = emphasize_recent_item
        self.use_date = use_date
        self.max_history_len = max_history_len
        self.history_text_max_length = history_text_max_length
        self.use_reviews = use_reviews
        self.days_filter = days_filter
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.include_target_date = include_target_date
        self.use_sasrec = use_sasrec
        self.sasrec_top_k = sasrec_top_k
        self.use_relative_date = use_relative_date
        self.trigger_items = trigger_items or {}
        self.trigger_emphasis_text = trigger_emphasis_text
        
        # Select the appropriate prompt template based on the dataset
        if data_name == 'yelp':
            self.templates = PROMPT_TEMPLATES_YELP
            print(f"✓ Using PROMPT_TEMPLATES_YELP for data_name='{data_name}'")
        else:
            self.templates = PROMPT_TEMPLATES
        
        # Set prompt type
        if prompt_type not in self.templates:
            print(f"⚠️  Unknown prompt type '{prompt_type}'. Available types: {list(self.templates.keys())}")
            print(f"   Using default 'recent_preference' type.")
            self.prompt_type = 'recent_preference'
        else:
            self.prompt_type = prompt_type
            print(f"✓ Using prompt type: '{self.prompt_type}'")
        
        # Load user2reviews_with_date.json
        self.user_reviews_with_date = {}
        if data_name:
            date_file_path = f"data/{data_name}/user2reviews_with_date.json"
            if os.path.exists(date_file_path):
                print(f"Loading date information from: {date_file_path}")
                with open(date_file_path, 'r') as f:
                    self.user_reviews_with_date = json.load(f)
                print(f"✓ Loaded date information for {len(self.user_reviews_with_date)} users")
            else:
                print(f"⚠️  Date file not found: {date_file_path}. Dates will not be included.")
                self.use_date = False
        
        # Load SASRec recommendation results
        self.sasrec_predictions = {}
        if self.use_sasrec and data_name:
            print(f"🔍 SASRec recommendations will be loaded per split in RecommendationDataset")
    
    def generate_prompt(self, item_ids: List[int], user_id: Optional[int] = None, target_item_id: Optional[int] = None, sasrec_items: Optional[List[int]] = None, trigger_item_id: Optional[int] = None) -> str:
        """
        Generate a prompt from a user sequence.

        Args:
            item_ids: list of item IDs
            user_id: user ID (used to look up date information, optional)
            target_timestamp: target timestamp (baseline for days_filter, optional)
            target_item_id: target/label item ID (used for including target date, optional)
            sasrec_items: list of SASRec recommended item IDs (optional)
            trigger_item_id: trigger item ID (item to emphasize, optional)

        Returns:
            generated prompt string
        """
        # Retrieve the user's review information
        user_reviews = []
        if user_id is not None:
            user_id_str = str(user_id)
            user_reviews = self.user_reviews_with_date.get(user_id_str, [])
        
        # Build a review mapping keyed by item ID
        item_to_review = {}
        if user_reviews:
            for review in user_reviews:
                item_id = int(review.get('item_id', -1))
                if item_id != -1:
                    item_to_review[item_id] = review
        
        # History text list
        history_text_list = []
        
        # Get target timestamp (used for relative date calculation and days_filter)
        target_timestamp = 0
        if target_item_id is not None and target_item_id in item_to_review:
            target_timestamp = int(item_to_review[target_item_id].get('timestamp', 0))
        
        # Process each item
        for idx, item_id in enumerate(item_ids):
            # print(type(item_id))
            item_data = self.item_metadata.get(item_id)
            if item_data is None:
                # Skip if metadata is not available
                print(f"⚠️  Item metadata not found for item {item_id}")
                continue

            # Time filtering (when days_filter is set and target_timestamp is provided)
            if self.days_filter is not None and target_timestamp > 0:
                review = item_to_review.get(item_id)
                if review:
                    timestamp = int(review.get('timestamp', 0))
                    if target_timestamp - timestamp > self.days_filter * 24 * 60 * 60:
                        # print(f"⚠️  Item timestamp is too old for item {item_id}")
                        continue
            
            item_title = item_data.get('title', 'Unknown Item')
            item_brand = item_data.get('brand', 'Unknown Brand')
            item_categories = item_data.get('category', 'Unknown Category')
            item_description = item_data.get('description', '')
        
            item_history_text = ""
            # Add date information
            if self.use_date and item_id in item_to_review:
                if self.use_relative_date and target_timestamp > 0:
                    # Calculate relative date (D-N format)
                    item_timestamp = int(item_to_review[item_id].get('timestamp', 0))
                    if item_timestamp > 0:
                        days_diff = (target_timestamp - item_timestamp) // (24 * 60 * 60)
                        item_history_text += f"Date: (D-{days_diff})\n"
                else:
                    # Show absolute date
                    item_date = item_to_review[item_id].get('date', '')
                    if item_date:
                        item_history_text += f"Date: {item_date}\n"
            
            # Base history format
            item_history_text += f"Title: {item_title}\n"
            
            if self.use_brand:
                item_history_text += f"Brand: {item_brand}\n"
            
            if self.use_category:
                item_history_text += f"Categories: {item_categories}\n"
            
            if self.use_description and item_description:
                item_description = item_description.replace("\n", " ")
                if len(item_description.split()) > self.history_text_max_length:
                    item_description = " ".join(
                        item_description.split()[:self.history_text_max_length]
                    ) + "..."
                item_history_text += f"Description: {item_description}\n"
            
            # Add review text
            if self.use_reviews and item_id in item_to_review:
                review_text = item_to_review[item_id].get('text', '')
                # limit review text words
                if review_text and len(review_text.split()) > self.history_text_max_length:
                    review_text = " ".join(review_text.split()[:self.history_text_max_length])
                if review_text:
                    item_history_text += f"Review:\n{review_text}\n"
            
            # Add trigger item emphasis text
            if trigger_item_id is not None and item_id == trigger_item_id and self.trigger_emphasis_text:
                item_history_text += f"**Note:** {self.trigger_emphasis_text}\n"
            
            history_text_list.append(item_history_text)
        
        # If history is empty, include at least the last item
        if len(history_text_list) == 0 and len(item_ids) > 0:
            last_item_id = item_ids[-1]

            item_title = self.item_metadata.get(last_item_id, {}).get('title', 'Unknown Item')
            item_brand = self.item_metadata.get(last_item_id, {}).get('brand', 'Unknown Brand')
            item_categories = self.item_metadata.get(last_item_id, {}).get('category', 'Unknown Category')
            item_description = self.item_metadata.get(last_item_id, {}).get('description', '')

            item_history_text = ""
            if self.use_date and last_item_id in item_to_review:
                if self.use_relative_date and target_timestamp > 0:
                    # Calculate relative date (D-N format)
                    item_timestamp = int(item_to_review[last_item_id].get('timestamp', 0))
                    if item_timestamp > 0:
                        days_diff = (target_timestamp - item_timestamp) // (24 * 60 * 60)
                        item_history_text += f"Date: (D-{days_diff})\n"
                else:
                    # Show absolute date
                    item_date = item_to_review[last_item_id].get('date', '')
                    if item_date:
                        item_history_text += f"Date: {item_date}\n"

            item_history_text += f"Title: {item_title}\n"
            if self.use_brand:
                item_history_text += f"Brand: {item_brand}\n"
            if self.use_category:
                item_history_text += f"Categories: {item_categories}\n"
            
            if self.use_reviews and last_item_id in item_to_review:
                review_text = item_to_review[last_item_id].get('text', '')
                if review_text and len(review_text.split()) > self.history_text_max_length:
                    review_text = " ".join(review_text.split()[:self.history_text_max_length])
                if review_text:
                    item_history_text += f"Review:\n{review_text}\n"
            
            history_text_list.append(item_history_text)
        
        # Apply history length constraint
        if len(history_text_list) > self.max_history_len:
            history_text_list = history_text_list[-self.max_history_len:]
        
        # Build final history text
        history_text = "\n\n".join(
            f"{i+1}. {history}" for i, history in enumerate(history_text_list)
        )
        
        # Emphasize the last item
        if self.use_last_item and len(item_ids) > 0:
            last_item = self.item_metadata.get(item_ids[-1], {})
            last_item_title = last_item.get('title', 'Unknown Item')
            history_text += f"\n\n`{last_item_title}` is the most recently purchased item."
        
        # Get the selected prompt template
        template = self.templates[self.prompt_type]
        
        # Add target item date
        target_date = ""
        if self.include_target_date and target_item_id is not None:
            if target_item_id in item_to_review:
                target_date = item_to_review[target_item_id].get('date', '')
            
            if target_date:
                target_date = f"- **Target Purchase Date:**: {target_date}\n"
        
        # Build SASRec recommendation result section
        sasrec_section = ""
        if self.use_sasrec and sasrec_items and len(sasrec_items) > 0:
            # Check if sasrec_section exists in the prompt template
            template = self.templates[self.prompt_type]
            if 'sasrec_section' in template:
                sasrec_section = template['sasrec_section']
                
                # Convert SASRec recommended item information to text
                sasrec_text_list = []
                for idx, item_id in enumerate(sasrec_items[:self.sasrec_top_k]):
                    item_data = self.item_metadata.get(item_id)
                    if item_data is None:
                        continue
                    
                    item_title = item_data.get('title', 'Unknown Item')
                    # limit title length to 100 words
                    item_title = " ".join(item_title.split()[:100])
                    
                    sasrec_item_text = f"{idx+1}. {item_title}"
                    sasrec_text_list.append(sasrec_item_text)
                
                if sasrec_text_list:
                    sasrec_section += "\n" + "\n".join(sasrec_text_list) + "\n\n"
        
        # Build final prompt
        prompt = (
            f"{template['head']}\n\n"
            f"{target_date}\n"
            f"- **User Purchase History:**\n"
            f"{history_text}\n"
            f"{sasrec_section}"
            f"{template['tail']}\n"
        )
        
        # Apply chat template
        if self.apply_chat_template and self.tokenizer is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        return prompt


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation systems.
    Contains user history, target items, and prompts.
    """
    
    def __init__(
        self,
        data_name: str,
        item_metadata: Dict,
        prompt_generator: PromptGenerator,
        split: str = "train",
        num_negs: int = 0,
        num_items: Optional[int] = None,
    ):
        """
        Args:
            data_name: dataset name
            item_metadata: item metadata dictionary
            prompt_generator: prompt generator
            split: data split ("train", "valid", "test")
            num_negs: number of negative items to pre-sample (0 disables pre-sampling)
            num_items: total number of items (required for negative sampling)
        """
        self.data_name = data_name
        self.item_metadata = item_metadata
        self.prompt_generator = prompt_generator
        self.split = split
        self.num_negs = num_negs
        self.num_items = num_items
        
        sequential_file = f"data/{data_name}/sequential_data.txt"
        self._load_real_data(sequential_file, split)
        
        # Load SASRec recommendation results (only when use_sasrec is True)
        self.sasrec_predictions = {}
        if self.prompt_generator.use_sasrec:
            sasrec_file = f"sasrec_results/SASRec_{data_name}_{split}_topk_prediction.json"
            if os.path.exists(sasrec_file):
                print(f"📦 Loading SASRec predictions from: {sasrec_file}")
                with open(sasrec_file, 'r') as f:
                    sasrec_data = json.load(f)
                    # Convert keys to int and extract only item IDs (first element of each [item_id, score] pair)
                    self.sasrec_predictions = {
                        int(k): [item[0] for item in v] 
                        for k, v in sasrec_data.items()
                    }
                print(f"✓ Loaded SASRec predictions for {len(self.sasrec_predictions)} users")
            else:
                print(f"⚠️  SASRec prediction file not found: {sasrec_file}. SASRec recommendations will not be included.")
                self.prompt_generator.use_sasrec = False
        
        # Pre-generate prompts at initialization
        print(f"✍️  Pre-generating prompts for {len(self.user_ids)} users...")
        self.prompt_dict = {}
        for user_id in self.user_ids:
            history = self.history_dict[user_id]
            target_item_id = self.target_dict[user_id]
            
            # Get SASRec recommendation results (if available)
            sasrec_items = self.sasrec_predictions.get(user_id, []) if self.prompt_generator.use_sasrec else None
            
            # Get trigger item (if available)
            trigger_item_id = self.prompt_generator.trigger_items.get(user_id, None)
            
            self.prompt_dict[user_id] = self.prompt_generator.generate_prompt(
                history, 
                user_id=user_id, 
                target_item_id=target_item_id,
                sasrec_items=sasrec_items,
                trigger_item_id=trigger_item_id,
            )

        # print sample prompts
        for user_id in [10]:
            print(f"User {user_id}: \n{self.prompt_dict[user_id]}")
            print("-" * 100)
        
        # Pre-sample negative items at initialization
        if self.num_negs > 0:
            if self.num_items is None:
                raise ValueError("num_items must be provided when num_negs > 0")
            self._load_negative_items()
        
        print(f"✓ {split.upper()} Dataset loaded: {len(self.user_ids)} users")
    
    def _load_real_data(
        self,
        sequential_file: str,
        split: str,
    ):
        """Load actual data"""
        all_user_ids = []
        all_history = {}
        all_targets = {}
        
        if split == "train":
            target_index = -3
        elif split == "valid":
            target_index = -2
        elif split == "test":
            target_index = -1
        else:
            raise ValueError(f"Invalid split: {split}")
        
        with open(sequential_file, "r") as f:
            for line in f:
                parts = [int(p) for p in line.strip().split()]
                user_id = parts[0]
                history = parts[1:target_index]
                target = parts[target_index]
                
                all_user_ids.append(user_id)
                all_history[user_id] = history
                all_targets[user_id] = target
        
        self.user_ids = all_user_ids
        self.history_dict = all_history
        self.target_dict = all_targets
    
    def filter_by_rank(self, csv_path: str, rank_min: Optional[int] = None, rank_max: Optional[int] = None):
        """
        Filter the dataset based on a rank range from an evaluation result CSV.

        Args:
            csv_path: path to the evaluation result CSV file (must contain user_id and rank columns)
            rank_min: minimum rank (None means no lower bound)
            rank_max: maximum rank (None means no upper bound)
        """
        # if not os.path.exists(csv_path):
        #     raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"\n{'='*80}")
        print(f"🔍 Filtering dataset by rank range")
        print(f"{'='*80}")
        print(f"  CSV file: {csv_path}")
        print(f"  Rank range: [{rank_min if rank_min is not None else 'None'}, {rank_max if rank_max is not None else 'None'}]")
        print(f"  Original size: {len(self.user_ids)} users")
        
        # Load CSV
        df = pd.read_csv(csv_path)

        # Verify required columns
        if 'user_id' not in df.columns or 'rank' not in df.columns:
            raise ValueError(f"CSV must contain 'user_id' and 'rank' columns. Found: {df.columns.tolist()}")
        
        # Filter by rank range
        mask = pd.Series([True] * len(df))
        if rank_min is not None:
            mask &= (df['rank'] >= rank_min)
        if rank_max is not None:
            mask &= (df['rank'] <= rank_max)
        
        filtered_df = df[mask]
        
        # Set of filtered user_ids
        filtered_user_ids = set(filtered_df['user_id'].tolist())
        
        print(f"  Filtered users from CSV: {len(filtered_user_ids)} users")
        
        # Filter dataset
        original_count = len(self.user_ids)
        self.user_ids = [uid for uid in self.user_ids if uid in filtered_user_ids]
        
        # Filter history and targets as well
        filtered_history = {uid: hist for uid, hist in self.history_dict.items() if uid in filtered_user_ids}
        filtered_target = {uid: tgt for uid, tgt in self.target_dict.items() if uid in filtered_user_ids}
        
        self.history_dict = filtered_history
        self.target_dict = filtered_target
        
        # Filter prompts as well (if already generated)
        if hasattr(self, 'prompt_dict'):
            self.prompt_dict = {uid: prompt for uid, prompt in self.prompt_dict.items() if uid in filtered_user_ids}
        
        # Filter negative items as well (if present)
        if hasattr(self, 'neg_items_dict'):
            self.neg_items_dict = {uid: items for uid, items in self.neg_items_dict.items() if uid in filtered_user_ids}
        
        print(f"  Filtered size: {len(self.user_ids)} users (removed {original_count - len(self.user_ids)} users)")
        print(f"{'='*80}\n")
    
    def _load_negative_items(self):
        """Pre-sample negative items per user"""
        negative_file = Path("data") / self.data_name / "negative.txt"
        if not negative_file.exists():
            raise FileNotFoundError(f"Negative pool file not found: {negative_file}")
        
        print(f"📦 Loading negative pool from: {negative_file}")
        negative_pool = {}
        
        with open(negative_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                user_id = int(parts[0])
                neg_items = [int(item_id) for item_id in parts[1:]]
                negative_pool[user_id] = neg_items[:self.num_negs]

        self.neg_items_dict = negative_pool
        
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        history = self.history_dict[user_id]
        target = self.target_dict[user_id]
        
        # Use pre-generated prompt
        prompt = self.prompt_dict[user_id]
        
        result = {
            "prompt": prompt,
            "history": history,
            "target": target,
            "user_id": user_id,
        }
        
        # Add negative items if present
        if self.num_negs > 0:
            result["neg_items"] = self.neg_items_dict[user_id]
        
        return result


def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    return {
        "queries": [item["query"] for item in batch],
        "histories": [item["history"] for item in batch],
        "targets": [item["target"] for item in batch],
        "user_ids": [item["user_id"] for item in batch],
    }


def load_item_metadata(data_name: str, data_dir: str = "data") -> Dict:
    """
    Load item metadata.

    Args:
        data_name: dataset name (e.g., "beauty")
        data_dir: data directory

    Returns:
        item metadata dictionary
    """
    # Try metadata file paths
    possible_paths = [
        f"{data_dir}/{data_name}/meta_text_fix.json",
    ]
    
    item_metadata = {}
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading item metadata from: {path}")
            data = json.load(open(path, 'r'))
            item_metadata = {int(k): v for k, v in data.items()}
            return item_metadata
    
    # Warn if metadata file was not found
    print(f"⚠️  Item metadata file not found. Using dummy metadata.")
    return {}


def create_dataloaders(
    args: argparse.Namespace,
    tokenizer: Optional = None,
    apply_chat_template: bool = True,
    trigger_items_train: Optional[Dict[int, int]] = None,
    trigger_items_valid: Optional[Dict[int, int]] = None,
    trigger_items_test: Optional[Dict[int, int]] = None,
    trigger_emphasis_text: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, PromptGenerator, Dict]:
    """
    Create Train/Valid/Test DataLoaders
    
    Args:
        args: argparse.Namespace
        tokenizer: Tokenizer instance
        apply_chat_template: Whether to apply chat template
        trigger_items_train: {user_id: trigger_item_id} for train split
        trigger_items_valid: {user_id: trigger_item_id} for valid split
        trigger_items_test: {user_id: trigger_item_id} for test split
        trigger_emphasis_text: Text to emphasize trigger item
    
    Returns:
        (train_dataloader, valid_dataloader, test_dataloader, prompt_generator, item_metadata)
    """
    # Load item metadata
    print(f"📦 Loading item metadata...")
    item_metadata = load_item_metadata(args.data_name)
    
    # Get num_items (use from args if available, otherwise infer from metadata)
    num_items = getattr(args, 'num_items', None)
    if num_items is None and len(item_metadata) > 0:
        num_items = max(item_metadata.keys()) + 1
        print(f"  Inferred num_items from metadata: {num_items}")
    
    # Get num_negs (use from args if available, otherwise default to 0)
    num_negs = getattr(args, 'num_negs', 0)
    
    # Prompt generator
    print(f"✍️  Creating prompt generator...")
    
    # Get use_date parameter (use from args if available, otherwise default to True)
    use_date = getattr(args, 'use_date', True)
    
    # Get prompt_type parameter (use from args if available, otherwise default to 'seq_rec')
    prompt_type = getattr(args, 'prompt_type', 'seq_rec')
    
    # Get emphasize_recent_item parameter (use from args if available, otherwise default to False)
    emphasize_recent_item = getattr(args, 'emphasize_recent_item', False)
    
    # Get include_target_date parameter (use from args if available, otherwise default to False)
    include_target_date = getattr(args, 'include_target_date', False)
    
    # Get use_sasrec parameter (use from args if available, otherwise default to False)
    use_sasrec = getattr(args, 'use_sasrec', False)
    
    # Get sasrec_top_k parameter (use from args if available, otherwise default to 5)
    sasrec_top_k = getattr(args, 'sasrec_top_k', 5)
    
    # Get days_filter parameter (use from args if available, otherwise default to None)
    days_filter = getattr(args, 'days_filter', None)
    
    # Get use_relative_date parameter (use from args if available, otherwise default to False)
    use_relative_date = getattr(args, 'use_relative_date', False)
    
    # Create separate prompt generators for each split to handle trigger_items
    prompt_generator_train = PromptGenerator(
        item_metadata=item_metadata,
        data_name=args.data_name,
        prompt_type=prompt_type,
        use_brand=args.use_brand,
        use_category=args.use_category,
        use_description=args.use_description,
        use_date=use_date,
        max_history_len=args.max_history_len,
        history_text_max_length=args.history_text_max_length,
        days_filter=days_filter,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        emphasize_recent_item=emphasize_recent_item,
        include_target_date=include_target_date,
        use_sasrec=use_sasrec,
        sasrec_top_k=sasrec_top_k,
        use_relative_date=use_relative_date,
        trigger_items=trigger_items_train,
        trigger_emphasis_text=trigger_emphasis_text,
    )
    
    prompt_generator_valid = PromptGenerator(
        item_metadata=item_metadata,
        data_name=args.data_name,
        prompt_type=prompt_type,
        use_brand=args.use_brand,
        use_category=args.use_category,
        use_description=args.use_description,
        use_date=use_date,
        max_history_len=args.max_history_len,
        history_text_max_length=args.history_text_max_length,
        days_filter=days_filter,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        emphasize_recent_item=emphasize_recent_item,
        include_target_date=include_target_date,
        use_sasrec=use_sasrec,
        sasrec_top_k=sasrec_top_k,
        use_relative_date=use_relative_date,
        trigger_items=trigger_items_valid,
        trigger_emphasis_text=trigger_emphasis_text,
    )
    
    prompt_generator_test = PromptGenerator(
        item_metadata=item_metadata,
        data_name=args.data_name,
        prompt_type=prompt_type,
        use_brand=args.use_brand,
        use_category=args.use_category,
        use_description=args.use_description,
        use_date=use_date,
        max_history_len=args.max_history_len,
        history_text_max_length=args.history_text_max_length,
        days_filter=days_filter,
        tokenizer=tokenizer,
        apply_chat_template=apply_chat_template,
        emphasize_recent_item=emphasize_recent_item,
        include_target_date=include_target_date,
        use_sasrec=use_sasrec,
        sasrec_top_k=sasrec_top_k,
        use_relative_date=use_relative_date,
        trigger_items=trigger_items_test,
        trigger_emphasis_text=trigger_emphasis_text,
    )
    
    # Create datasets
    print(f"📊 Creating datasets...")
    
    # if args.num_epochs > 0:
        # Train dataset
    train_dataset = RecommendationDataset(
        data_name=args.data_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator_train,
        split="train",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # Filter train dataset (rank-range based)
    filter_train_csv = getattr(args, 'filter_train_csv', None)
    if filter_train_csv is not None:
        rank_min = getattr(args, 'rank_min', None)
        rank_max = getattr(args, 'rank_max', None)
        train_dataset.filter_by_rank(filter_train_csv, rank_min, rank_max)
    
    # Valid dataset
    valid_dataset = RecommendationDataset(
        data_name=args.data_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator_valid,
        split="valid",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # Filter valid dataset (optional)
    filter_valid_csv = getattr(args, 'filter_valid_csv', None)
    if filter_valid_csv is not None:
        rank_min = getattr(args, 'rank_min', None)
        rank_max = getattr(args, 'rank_max', None)
        valid_dataset.filter_by_rank(filter_valid_csv, rank_min, rank_max)
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    test_dataset = RecommendationDataset(
        data_name=args.data_name,
        item_metadata=item_metadata,
        prompt_generator=prompt_generator_test,
        split="test",
        num_negs=num_negs,
        num_items=num_items,
    )
    
    # Filter test dataset (optional)
    filter_test_csv = getattr(args, 'filter_test_csv', None)
    if filter_test_csv is not None:
        rank_min = getattr(args, 'rank_min', None)
        rank_max = getattr(args, 'rank_max', None)
        test_dataset.filter_by_rank(filter_test_csv, rank_min, rank_max)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"✓ DataLoaders created:")

    print(f"  Test samples: {len(test_dataset)}")
    if num_negs > 0:
        print(f"  Negative samples per user: {num_negs}")
    
    # Return test prompt_generator as the primary one
    return train_dataset, valid_dataset, test_dataset, prompt_generator_test, item_metadata
    # return train_dataloader, valid_dataloader, test_dataloader, prompt_generator, item_metadata

