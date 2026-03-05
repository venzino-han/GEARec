"""
Dataset utilities for RL4Rec
추천 시스템 학습을 위한 데이터셋 및 데이터로더 유틸리티
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
    사용자 시퀀스로부터 프롬프트를 생성하는 클래스
    
    지원하는 프롬프트 타입:
    - 'preference': 사용자 선호도 묘사
    - 'next_item': 다음 아이템 예측
    - 'recommendation': 추천 아이템 생성
    - 'user_profile': 사용자 프로필 생성
    - 'recent_preference': 최근 선호도 묘사
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
            item_metadata: 아이템 메타데이터 딕셔너리
            data_name: 데이터셋 이름 (날짜 정보 로드에 사용)
            prompt_type: 프롬프트 타입 ('preference', 'next_item', 'recommendation', 'user_profile', 'recent_preference', 'reasoning')
            use_brand: 브랜드 정보 포함 여부
            use_category: 카테고리 정보 포함 여부
            use_description: 설명 정보 포함 여부
            use_features: 특징 정보 포함 여부
            use_last_item: 마지막 아이템 강조 여부
            use_date: 날짜 정보 포함 여부 (히스토리 및 최근 구매 강조에 사용)
            max_history_len: 최대 히스토리 길이
            history_text_max_length: 히스토리 텍스트 최대 단어 수 (review text에도 적용)
            use_reviews: 리뷰 텍스트 포함 여부
            days_filter: 최근 N일 이내의 리뷰만 포함 (None이면 필터링 안함)
            tokenizer: 토크나이저 (챗 템플릿 적용에 필요, 선택적)
            apply_chat_template: 챗 템플릿 적용 여부
            emphasize_recent_item: 최근 구매 아이템을 상세하게 강조할지 여부 ("This user's most recent purchase is..." 형식, use_date가 True면 구매 날짜도 포함)
            include_target_date: 타겟/레이블 아이템의 구매 날짜를 프롬프트 마지막에 포함할지 여부
            use_sasrec: SASRec 추천 결과를 프롬프트에 포함할지 여부
            sasrec_top_k: SASRec 추천 결과에서 상위 K개 아이템만 포함
            use_relative_date: 상대 날짜 표기 사용 여부 (True면 타겟 날짜 기준으로 "(D-10)", "(D-20)" 형식으로 표시)
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
        
        # 데이터셋에 따라 적절한 프롬프트 템플릿 선택
        if data_name == 'yelp':
            self.templates = PROMPT_TEMPLATES_YELP
            print(f"✓ Using PROMPT_TEMPLATES_YELP for data_name='{data_name}'")
        else:
            self.templates = PROMPT_TEMPLATES
        
        # 프롬프트 타입 설정
        if prompt_type not in self.templates:
            print(f"⚠️  Unknown prompt type '{prompt_type}'. Available types: {list(self.templates.keys())}")
            print(f"   Using default 'recent_preference' type.")
            self.prompt_type = 'recent_preference'
        else:
            self.prompt_type = prompt_type
            print(f"✓ Using prompt type: '{self.prompt_type}'")
        
        # user2reviews_with_date.json 로드
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
        
        # SASRec 추천 결과 로드
        self.sasrec_predictions = {}
        if self.use_sasrec and data_name:
            print(f"🔍 SASRec recommendations will be loaded per split in RecommendationDataset")
    
    def generate_prompt(self, item_ids: List[int], user_id: Optional[int] = None, target_item_id: Optional[int] = None, sasrec_items: Optional[List[int]] = None, trigger_item_id: Optional[int] = None) -> str:
        """
        사용자 시퀀스로부터 프롬프트 생성
        
        Args:
            item_ids: 아이템 ID 리스트
            user_id: 사용자 ID (날짜 정보 조회용, 선택적)
            target_timestamp: 타겟 타임스탬프 (days_filter 적용시 기준, 선택적)
            target_item_id: 타겟/레이블 아이템 ID (타겟 날짜 포함용, 선택적)
            sasrec_items: SASRec 추천 아이템 ID 리스트 (선택적)
            trigger_item_id: Trigger 아이템 ID (강조할 아이템, 선택적)
        
        Returns:
            생성된 프롬프트 문자열
        """
        # 사용자의 리뷰 정보 가져오기
        user_reviews = []
        if user_id is not None:
            user_id_str = str(user_id)
            user_reviews = self.user_reviews_with_date.get(user_id_str, [])
        
        # 아이템 ID를 키로 하는 리뷰 매핑 생성
        item_to_review = {}
        if user_reviews:
            for review in user_reviews:
                item_id = int(review.get('item_id', -1))
                if item_id != -1:
                    item_to_review[item_id] = review
        
        # 히스토리 텍스트 리스트
        history_text_list = []
        
        # 타겟 타임스탬프 가져오기 (상대 날짜 계산 및 days_filter에 사용)
        target_timestamp = 0
        if target_item_id is not None and target_item_id in item_to_review:
            target_timestamp = int(item_to_review[target_item_id].get('timestamp', 0))
        
        # 각 아이템 처리
        for idx, item_id in enumerate(item_ids):
            # print(type(item_id))
            item_data = self.item_metadata.get(item_id)
            if item_data is None:
                # 메타데이터가 없는 경우 스킵
                print(f"⚠️  Item metadata not found for item {item_id}")
                continue

            # 시간 필터링 (days_filter가 설정되어 있고 target_timestamp가 주어진 경우)
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
            # 날짜 정보 추가
            if self.use_date and item_id in item_to_review:
                if self.use_relative_date and target_timestamp > 0:
                    # 상대 날짜 계산 (D-N 형식)
                    item_timestamp = int(item_to_review[item_id].get('timestamp', 0))
                    if item_timestamp > 0:
                        days_diff = (target_timestamp - item_timestamp) // (24 * 60 * 60)
                        item_history_text += f"Date: (D-{days_diff})\n"
                else:
                    # 절대 날짜 표시
                    item_date = item_to_review[item_id].get('date', '')
                    if item_date:
                        item_history_text += f"Date: {item_date}\n"
            
            # 기본 히스토리 포맷
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
            
            # 리뷰 텍스트 추가
            if self.use_reviews and item_id in item_to_review:
                review_text = item_to_review[item_id].get('text', '')
                # limit review text words
                if review_text and len(review_text.split()) > self.history_text_max_length:
                    review_text = " ".join(review_text.split()[:self.history_text_max_length])
                if review_text:
                    item_history_text += f"Review:\n{review_text}\n"
            
            # Trigger item 강조 텍스트 추가
            if trigger_item_id is not None and item_id == trigger_item_id and self.trigger_emphasis_text:
                item_history_text += f"**Note:** {self.trigger_emphasis_text}\n"
            
            history_text_list.append(item_history_text)
        
        # 히스토리가 비어있는 경우 마지막 아이템이라도 포함
        if len(history_text_list) == 0 and len(item_ids) > 0:
            last_item_id = item_ids[-1]

            item_title = self.item_metadata.get(last_item_id, {}).get('title', 'Unknown Item')
            item_brand = self.item_metadata.get(last_item_id, {}).get('brand', 'Unknown Brand')
            item_categories = self.item_metadata.get(last_item_id, {}).get('category', 'Unknown Category')
            item_description = self.item_metadata.get(last_item_id, {}).get('description', '')

            item_history_text = ""
            if self.use_date and last_item_id in item_to_review:
                if self.use_relative_date and target_timestamp > 0:
                    # 상대 날짜 계산 (D-N 형식)
                    item_timestamp = int(item_to_review[last_item_id].get('timestamp', 0))
                    if item_timestamp > 0:
                        days_diff = (target_timestamp - item_timestamp) // (24 * 60 * 60)
                        item_history_text += f"Date: (D-{days_diff})\n"
                else:
                    # 절대 날짜 표시
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
        
        # 히스토리 길이 제약 적용
        if len(history_text_list) > self.max_history_len:
            history_text_list = history_text_list[-self.max_history_len:]
        
        # 최종 히스토리 텍스트 구성
        history_text = "\n\n".join(
            f"{i+1}. {history}" for i, history in enumerate(history_text_list)
        )
        
        # 마지막 아이템 강조
        if self.use_last_item and len(item_ids) > 0:
            last_item = self.item_metadata.get(item_ids[-1], {})
            last_item_title = last_item.get('title', 'Unknown Item')
            history_text += f"\n\n`{last_item_title}` is the most recently purchased item."
        
        # 선택된 프롬프트 템플릿 가져오기
        template = self.templates[self.prompt_type]
        
        # 타겟 아이템 날짜 추가
        target_date = ""
        if self.include_target_date and target_item_id is not None:
            if target_item_id in item_to_review:
                target_date = item_to_review[target_item_id].get('date', '')
            
            if target_date:
                target_date = f"- **Target Purchase Date:**: {target_date}\n"
        
        # SASRec 추천 결과 섹션 생성
        sasrec_section = ""
        if self.use_sasrec and sasrec_items and len(sasrec_items) > 0:
            # 프롬프트 템플릿에서 sasrec_section이 있는지 확인
            template = self.templates[self.prompt_type]
            if 'sasrec_section' in template:
                sasrec_section = template['sasrec_section']
                
                # SASRec 추천 아이템들의 정보를 텍스트로 변환
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
        
        # 최종 프롬프트 생성
        prompt = (
            f"{template['head']}\n\n"
            f"{target_date}\n"
            f"- **User Purchase History:**\n"
            f"{history_text}\n"
            f"{sasrec_section}"
            f"{template['tail']}\n"
        )
        
        # 챗 템플릿 적용
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
    추천 시스템용 데이터셋
    사용자 히스토리, 타겟 아이템, 프롬프트를 포함
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
            data_name: 데이터셋 이름
            item_metadata: 아이템 메타데이터 딕셔너리
            prompt_generator: 프롬프트 생성기
            split: 데이터 분할 ("train", "valid", "test")
            num_negs: 사전 샘플링할 negative 아이템 수 (0이면 비활성화)
            num_items: 전체 아이템 수 (negative sampling에 필요)
        """
        self.data_name = data_name
        self.item_metadata = item_metadata
        self.prompt_generator = prompt_generator
        self.split = split
        self.num_negs = num_negs
        self.num_items = num_items
        
        sequential_file = f"data/{data_name}/sequential_data.txt"
        self._load_real_data(sequential_file, split)
        
        # SASRec 추천 결과 로드 (use_sasrec이 True인 경우에만)
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
        
        # 프롬프트 미리 생성 (초기화 시점)
        print(f"✍️  Pre-generating prompts for {len(self.user_ids)} users...")
        self.prompt_dict = {}
        for user_id in self.user_ids:
            history = self.history_dict[user_id]
            target_item_id = self.target_dict[user_id]
            
            # SASRec 추천 결과 가져오기 (있는 경우)
            sasrec_items = self.sasrec_predictions.get(user_id, []) if self.prompt_generator.use_sasrec else None
            
            # Trigger item 가져오기 (있는 경우)
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
        
        # Negative items 미리 샘플링 (초기화 시점)
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
        """실제 데이터 로드"""
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
        평가 결과 CSV의 rank 범위를 기반으로 데이터셋 필터링
        
        Args:
            csv_path: 평가 결과 CSV 파일 경로 (user_id, rank 컬럼 포함)
            rank_min: 최소 rank (None이면 제한 없음)
            rank_max: 최대 rank (None이면 제한 없음)
        """
        # if not os.path.exists(csv_path):
        #     raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"\n{'='*80}")
        print(f"🔍 Filtering dataset by rank range")
        print(f"{'='*80}")
        print(f"  CSV file: {csv_path}")
        print(f"  Rank range: [{rank_min if rank_min is not None else 'None'}, {rank_max if rank_max is not None else 'None'}]")
        print(f"  Original size: {len(self.user_ids)} users")
        
        # CSV 로드
        df = pd.read_csv(csv_path)
        
        # 필수 컬럼 확인
        if 'user_id' not in df.columns or 'rank' not in df.columns:
            raise ValueError(f"CSV must contain 'user_id' and 'rank' columns. Found: {df.columns.tolist()}")
        
        # rank 범위 필터링
        mask = pd.Series([True] * len(df))
        if rank_min is not None:
            mask &= (df['rank'] >= rank_min)
        if rank_max is not None:
            mask &= (df['rank'] <= rank_max)
        
        filtered_df = df[mask]
        
        # 필터링된 user_id 세트
        filtered_user_ids = set(filtered_df['user_id'].tolist())
        
        print(f"  Filtered users from CSV: {len(filtered_user_ids)} users")
        
        # 데이터셋 필터링
        original_count = len(self.user_ids)
        self.user_ids = [uid for uid in self.user_ids if uid in filtered_user_ids]
        
        # 히스토리와 타겟도 필터링
        filtered_history = {uid: hist for uid, hist in self.history_dict.items() if uid in filtered_user_ids}
        filtered_target = {uid: tgt for uid, tgt in self.target_dict.items() if uid in filtered_user_ids}
        
        self.history_dict = filtered_history
        self.target_dict = filtered_target
        
        # 프롬프트도 필터링 (이미 생성된 경우)
        if hasattr(self, 'prompt_dict'):
            self.prompt_dict = {uid: prompt for uid, prompt in self.prompt_dict.items() if uid in filtered_user_ids}
        
        # negative items도 필터링 (있는 경우)
        if hasattr(self, 'neg_items_dict'):
            self.neg_items_dict = {uid: items for uid, items in self.neg_items_dict.items() if uid in filtered_user_ids}
        
        print(f"  Filtered size: {len(self.user_ids)} users (removed {original_count - len(self.user_ids)} users)")
        print(f"{'='*80}\n")
    
    def _load_negative_items(self):
        """각 사용자별로 negative items 사전 샘플링"""
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
        
        # 미리 생성된 프롬프트 사용
        prompt = self.prompt_dict[user_id]
        
        result = {
            "prompt": prompt,
            "history": history,
            "target": target,
            "user_id": user_id,
        }
        
        # Negative items가 있으면 추가
        if self.num_negs > 0:
            result["neg_items"] = self.neg_items_dict[user_id]
        
        return result


def collate_fn(batch):
    """
    DataLoader용 collate function
    """
    return {
        "queries": [item["query"] for item in batch],
        "histories": [item["history"] for item in batch],
        "targets": [item["target"] for item in batch],
        "user_ids": [item["user_id"] for item in batch],
    }


def load_item_metadata(data_name: str, data_dir: str = "data") -> Dict:
    """
    아이템 메타데이터 로드
    
    Args:
        data_name: 데이터셋 이름 (e.g., "beauty")
        data_dir: 데이터 디렉토리
    
    Returns:
        아이템 메타데이터 딕셔너리
    """
    # 메타데이터 파일 경로 시도
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
    
    # 메타데이터 파일을 찾지 못한 경우 경고
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
    Train/Valid/Test DataLoader 생성
    
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
    # 아이템 메타데이터 로드
    print(f"📦 Loading item metadata...")
    item_metadata = load_item_metadata(args.data_name)
    
    # num_items 가져오기 (args에 있으면 사용, 없으면 메타데이터에서 추출)
    num_items = getattr(args, 'num_items', None)
    if num_items is None and len(item_metadata) > 0:
        num_items = max(item_metadata.keys()) + 1
        print(f"  Inferred num_items from metadata: {num_items}")
    
    # num_negs 가져오기 (args에 있으면 사용, 없으면 0)
    num_negs = getattr(args, 'num_negs', 0)
    
    # 프롬프트 생성기
    print(f"✍️  Creating prompt generator...")
    
    # use_date 파라미터 가져오기 (args에 있으면 사용, 없으면 True)
    use_date = getattr(args, 'use_date', True)
    
    # prompt_type 파라미터 가져오기 (args에 있으면 사용, 없으면 'recent_preference')
    prompt_type = getattr(args, 'prompt_type', 'seq_rec')
    
    # emphasize_recent_item 파라미터 가져오기 (args에 있으면 사용, 없으면 False)
    emphasize_recent_item = getattr(args, 'emphasize_recent_item', False)
    
    # include_target_date 파라미터 가져오기 (args에 있으면 사용, 없으면 False)
    include_target_date = getattr(args, 'include_target_date', False)
    
    # use_sasrec 파라미터 가져오기 (args에 있으면 사용, 없으면 False)
    use_sasrec = getattr(args, 'use_sasrec', False)
    
    # sasrec_top_k 파라미터 가져오기 (args에 있으면 사용, 없으면 5)
    sasrec_top_k = getattr(args, 'sasrec_top_k', 5)
    
    # days_filter 파라미터 가져오기 (args에 있으면 사용, 없으면 None)
    days_filter = getattr(args, 'days_filter', None)
    
    # use_relative_date 파라미터 가져오기 (args에 있으면 사용, 없으면 False)
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
    
    # 데이터셋 생성
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
    
    # Train dataset 필터링 (rank 범위 기반)
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
    
    # Valid dataset 필터링 (선택사항)
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
    
    # Test dataset 필터링 (선택사항)
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

