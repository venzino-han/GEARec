"""
Evaluation class for recommendation model assessment
"""
import gc
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.pooling_params import PoolingParams
from sentence_transformers import SentenceTransformer

from utils.reward_function import calculate_ndcg, calculate_hit_rate, extract_query_from_tags


def get_last_item_text(dataset, item_metadata, use_brand=True, use_category=True):
    """
    Convert the last purchased item information for each user into text.

    Args:
        dataset: RecommendationDataset instance
        item_metadata: item metadata
        use_brand: whether to include brand
        use_category: whether to include category

    Returns:
        last_item_texts: list of last item texts for each sample
    """
    last_item_texts = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        history = data.get('history', [])
        
        if len(history) > 0:
            last_item_id = history[-1]  # last item
            item_info = item_metadata.get(last_item_id, {})
            
            text_parts = []
            title = item_info.get('title', '')
            # limit title length to 64 words
            title = " ".join(title.split()[:64])
            text_parts.append(f"Last Item: {title}")
            
            if use_brand:
                brand = item_info.get('brand', '')
                if brand:
                    text_parts.append(f"Brand: {brand}")
            
            if use_category:
                category = item_info.get('category', '')
                if category:
                    text_parts.append(f"Category: {category}")
            
            last_item_text = "\n".join(text_parts)
        else:
            last_item_text = ""
        
        last_item_texts.append(last_item_text)
    
    return last_item_texts


class RecommendationEvaluator:
    """
    Class for evaluating recommendation models.
    Performs text generation and embedding-based retrieval using vLLM.
    """

    def __init__(self, args, checkpoint_dir, item_metadata=None):
        """
        Args:
            args: training configuration parameters
            checkpoint_dir: path to model checkpoint for evaluation
            item_metadata: item metadata (required when prepend_last_item is used)
        """
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_metadata = item_metadata
        
        # vLLM models (lazy loading)
        self.llm = None
        self.emb_llm = None
        self.item_embeddings = None
        self.use_sentence_transformers = False  # whether to use SentenceTransformer

        # For item popularity and Novelty computation
        self.item_popularity = None
        self.item_novelty = None
        self.hot_items = None  # top 20% popular items set
        self.cold_items = None  # remaining 80% items set

        self._load_item_embeddings()
        self._compute_item_popularity()
        # Cold/Hot classification
        
        
    def _load_generation_model(self):
        """Load vLLM model for text generation"""
        if hasattr(self.args, 'zeroshot_evaluation') and self.args.zeroshot_evaluation:
            print("🤖 Loading model with vLLM for zeroshot evaluation...")
            self.sampling_params = SamplingParams(
                temperature=0.01,
                max_tokens=self.args.eval_max_tokens,
                repetition_penalty=1.1,
            )
            self.llm = LLM(
                model=self.args.model_name,
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_length,
                max_num_seqs=self.args.eval_batch_size,
            )

        if self.llm is None:
            print("🤖 Loading model with vLLM for generation...")
            self.sampling_params = SamplingParams(
                # n=1,
                temperature=0.01,
                max_tokens=self.args.eval_max_tokens,
                repetition_penalty=1.1,
                # stop=["<|eot_id|>", "<|reserved_special_token_0|>", "<eos>"],
            )
        
            self.llm = LLM(
                model=self.checkpoint_dir,
                tensor_parallel_size=1,
                dtype=torch.bfloat16,
                # dtype=torch.bfloat16 if self.args.bf16 else torch.float32,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_length,
                max_num_seqs=self.args.eval_batch_size,
            )
    
    def _load_embedding_model(self):
        """Load vLLM or SentenceTransformer model for embedding computation"""
        if self.emb_llm is None:
            print("🔍 Loading embedding model for retrieval...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            use_sentence_transformers = getattr(self.args, 'use_sentence_transformers', False)

            # Check conditions for using SentenceTransformer
            if use_sentence_transformers:
                print("🔍 Using SentenceTransformer for embedding...")
                self.emb_llm = SentenceTransformer(emb_model_name)
                self.use_sentence_transformers = True
            else:
                print("🔍 Using vLLM for embedding...")
                # Set pooling parameters
                self.pooling_params = PoolingParams(
                    truncate_prompt_tokens=self.args.eval_emb_max_length,
                    task="embed",
                )
                
                self.emb_llm = LLM(
                    model=emb_model_name,
                    # task="embed",
                    runner="pooling",
                    enforce_eager=True,
                    gpu_memory_utilization=self.args.gpu_memory_utilization,
                    trust_remote_code=True,
                    max_model_len=self.args.eval_emb_max_length,
                    max_num_seqs=self.args.eval_emb_batch_size,
                )
                self.use_sentence_transformers = False
    
    def _load_item_embeddings(self):
        """Load item embeddings"""
        if self.item_embeddings is None:
            print("📦 Loading item embeddings...")
            emb_model_name = getattr(self.args, 'emb_model_name', 'mixedbread-ai/mxbai-embed-large-v1')
            emb_model_name_dir = emb_model_name.split('/')[-1]
            emb_type = getattr(self.args, 'emb_type', 'title')
            
            try:
                emb_file = f"data_emb/{self.args.data_name}_{emb_type}_{emb_model_name_dir}.pt"
                self.item_embeddings = torch.load(emb_file, map_location=self.device)
            except:
                emb_file = f"data_emb/{self.args.data_name}_{emb_type}_{emb_model_name_dir}_emb.pt"
                self.item_embeddings = torch.load(emb_file, map_location=self.device)
            self.item_embeddings = self.item_embeddings / self.item_embeddings.norm(dim=-1, keepdim=True)
            print(f"✓ Loaded item embeddings: {self.item_embeddings.shape}")
    
    def _compute_item_popularity(self):
        """
        Compute item popularity and Novelty (with caching support).
        Hot/Cold item split: top 20% popularity = Hot, remaining 80% = Cold.

        Reads from train set's sequential_data.txt file.

        Saved as 3 separate files:
        - item_popularity.npy: numpy array
        - item_novelty.npy: numpy array
        - item_cold_hot.json: {"cold_items": [...], "hot_items": [...]}
        """
        if self.item_popularity is None:
            # Cache file paths
            cache_dir = Path(f"./data/{self.args.data_name}")
            popularity_file = cache_dir / "item_popularity.npy"
            novelty_file = cache_dir / "item_novelty.npy"
            cold_hot_file = cache_dir / "item_cold_hot.json"

            # Load if all cache files exist
            if popularity_file.exists() and novelty_file.exists() and cold_hot_file.exists():
                print(f"📦 Loading cached item data from {cache_dir}...")
                try:
                    # Load popularity (.npy file)
                    self.item_popularity = np.load(popularity_file)

                    # Load novelty (.npy file)
                    self.item_novelty = np.load(novelty_file)

                    # Load Cold/Hot (JSON file)
                    with open(cold_hot_file, 'r') as f:
                        cold_hot_data = json.load(f)

                    # Convert Cold/Hot items to sets
                    self.cold_items = set(cold_hot_data['cold_items'])
                    self.hot_items = set(cold_hot_data['hot_items'])

                    max_pop = self.item_popularity.max()
                    print(f"✓ Item data loaded from cache. Max popularity: {max_pop:.4f}, Mean novelty: {self.item_novelty.mean():.4f}")
                    print(f"✓ Hot items: {len(self.hot_items)}, Cold items: {len(self.cold_items)}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load cache: {e}. Computing from scratch...")

            # Compute if no cache
            print("📊 Computing item popularity and novelty from sequential_data.txt...")

            # Total number of items
            num_items = self.item_embeddings.shape[0]

            # Read sequential_data.txt file
            sequential_file = Path(f"./data/{self.args.data_name}/sequential_data.txt")
            # Count occurrences for each item
            item_counts = np.zeros(num_items + 1, dtype=np.int32)  # 1-indexed

            with open(sequential_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # First element is user_id, rest is item sequence (history + target)
                    items = [int(x) for x in parts[1:-2]]

                    # Count occurrences of all items
                    for item_id in items:
                        item_counts[item_id] += 1

            print(f"✓ Loaded {sequential_file}, total items processed")

            # Store popularity
            self.item_popularity = (item_counts+1) / item_counts.sum()
            max_count = item_counts.max()
            self.item_novelty = -np.log(self.item_popularity)

            # Hot/Cold split: based on top 20% popularity threshold
            # Sort by popularity and compute top 20% threshold
            sorted_counts = np.sort(item_counts)[::-1]  # descending order
            threshold_idx = int(len(sorted_counts) * 0.2)
            threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else 0

            # Hot items: items with popularity >= threshold (1-indexed)
            # Cold items: items with popularity < threshold (1-indexed)
            hot_mask = (item_counts >= threshold) & (np.arange(len(item_counts)) > 0)
            cold_mask = (item_counts < threshold) & (np.arange(len(item_counts)) > 0)

            self.hot_items = set(np.where(hot_mask)[0].tolist())
            self.cold_items = set(np.where(cold_mask)[0].tolist())

            print(f"✓ Item popularity computed. Max count: {max_count}, Mean novelty: {self.item_novelty.mean():.4f}")
            print(f"✓ Hot items (top 20%): {len(self.hot_items)}, Cold items (80%): {len(self.cold_items)}, Threshold: {threshold}")

            # Save to cache files (3 separate files)
            try:
                # Create cache directory
                cache_dir.mkdir(parents=True, exist_ok=True)

                # 1. Save popularity (.npy file)
                np.save(popularity_file, self.item_popularity)
                print(f"💾 Item popularity cached to {popularity_file}")

                # 2. Save novelty (.npy file)
                np.save(novelty_file, self.item_novelty)
                print(f"💾 Item novelty cached to {novelty_file}")

                # 3. Save Cold/Hot (JSON file)
                cold_hot_data = {
                    'cold_items': list(self.cold_items),
                    'hot_items': list(self.hot_items),
                    'threshold': int(threshold),
                    'num_cold': len(self.cold_items),
                    'num_hot': len(self.hot_items),
                }
                with open(cold_hot_file, 'w') as f:
                    json.dump(cold_hot_data, f)
                print(f"💾 Cold/Hot items cached to {cold_hot_file}")
                
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")
    
    def _identify_cold_warm_items(self, targets):
        """
        Classify target items as Cold/Hot.
        Based on train data popularity: top 20% = Hot, remaining 80% = Cold.

        Args:
            targets: list of target item IDs

        Returns:
            cold_indices: list of Cold target indices
            hot_indices: list of Hot target indices
        """
        if self.hot_items is None or self.cold_items is None:
            raise ValueError("Hot/Cold items not computed. Call _compute_item_popularity first.")

        cold_indices = []
        hot_indices = []

        for i, target in enumerate(targets):
            # If target item is in hot_items set -> hot, otherwise -> cold
            if target in self.hot_items:
                hot_indices.append(i)
            else:
                cold_indices.append(i)
        
        return cold_indices, hot_indices
    
    def generate_all_texts(self, prompts):
        """
        Generate text for all prompts.

        Args:
            prompts: list of prompts to generate
            batch_size: batch size (uses args.eval_batch_size if None)
        
        Returns:
            generated_texts: list of generated texts
        """
        self._load_generation_model()
        # Sampling parameters are already set
        print(f"🚀 Generating responses for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts
    
    def compute_embeddings(self, texts):
        """
        Compute embeddings for all texts at once.
        If a <query> tag is present, only the text inside the tag is used.

        Args:
            texts: list of texts to embed
            batch_size: batch size (uses args.eval_emb_batch_size if None)

        Returns:
            query_embeddings: normalized query embeddings [num_texts, emb_dim]
        """
        self._load_embedding_model()

        # Extract <query> tag content if present, otherwise use original text
        processed_texts = [extract_query_from_tags(text, tag="query") for text in texts]

        print(f"🔍 Computing embeddings for {len(processed_texts)} texts...")

        if self.use_sentence_transformers:
            # Use SentenceTransformer
            print("🔍 Using SentenceTransformer encode...")
            query_embeddings = self.emb_llm.encode(
                processed_texts,
                batch_size=self.args.eval_emb_batch_size,
                show_progress_bar=True,
                convert_to_tensor=True
            )

        else:
            # Use vLLM
            # add cls token
            processed_texts_with_cls = [f"[CLS] {text}" for text in processed_texts]

            # Compute embeddings
            emb_outputs = self.emb_llm.encode(
                prompts=processed_texts_with_cls,
                pooling_task="embed",
                pooling_params=self.pooling_params,
                use_tqdm=True,
            )
            # Extract query embeddings
            embeddings_list = [
                torch.as_tensor(out.outputs.data, dtype=torch.float32, device=self.device)
                for out in emb_outputs
            ]
            query_embeddings = torch.stack(embeddings_list)
        
        return query_embeddings
    
    def compute_retrieval_metrics(self, query_embeddings, targets, histories, ks=[5, 10, 20]):
        """
        Compute retrieval metrics (processed in batches).
        Includes Cold/Hot split and Novelty metrics.
        - Cold: bottom 80% popularity items
        - Hot: top 20% popularity items

        Args:
            query_embeddings: query embeddings [num_queries, emb_dim]
            targets: list of target item IDs
            histories: list of lists of history item IDs
            ks: k values to evaluate

        Returns:
            metrics: metrics dictionary
            rank_info: list of rank and score info for each sample
        """
        cold_indices, hot_indices = self._identify_cold_warm_items(targets)
        print(f"  Cold targets: {len(cold_indices)}, Hot targets: {len(hot_indices)}")
        self.cold_indices = cold_indices
        self.hot_indices = hot_indices
        print(f"📊 Computing retrieval metrics...")

        # Initialize metrics storage dictionary
        metrics = {f'ndcg@{k}': [] for k in ks}
        metrics.update({f'hit@{k}': [] for k in ks})
        metrics.update({f'cold_ndcg@{k}': [] for k in ks})
        metrics.update({f'cold_hit@{k}': [] for k in ks})
        metrics.update({f'hot_ndcg@{k}': [] for k in ks})
        metrics.update({f'hot_hit@{k}': [] for k in ks})
        metrics.update({f'novelty@{k}': [] for k in ks})
        metrics.update({f'novelty_hit@{k}': [] for k in ks})

        # List for storing rank information
        rank_info = []

        # Set batch size
        batch_size = self.args.eval_batch_size
        num_queries = query_embeddings.shape[0]
        num_batches = (num_queries + batch_size - 1) // batch_size
        
        print(f"Processing {num_queries} queries in {num_batches} batches...")
        
        for batch_idx in tqdm(range(num_batches), desc="Computing metrics"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_queries)
            
            # Extract batch data
            batch_query_emb = query_embeddings[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            batch_histories = histories[start_idx:end_idx]

            # Compute similarity with all items
            batch_scores = torch.matmul(batch_query_emb, self.item_embeddings.T)  # [batch_size, num_items]

            # Exclude previously purchased items
            for i in range(len(batch_targets)):
                history_indices = [idx for idx in batch_histories[i] if idx != batch_targets[i]]
                if history_indices:
                    batch_scores[i, history_indices] = -float('inf')

            # Compute rank and Novelty metrics for each sample
            for i in range(len(batch_targets)):
                global_idx = start_idx + i
                target_item = batch_targets[i]
                target_score = batch_scores[i, target_item].item()

                # Number of items with a higher score than target = rank - 1
                rank = (batch_scores[i] > target_score).sum().item() + 1

                # Cold/Warm classification
                is_cold = global_idx in self.cold_indices
                
                rank_info.append({
                    'target_item': target_item,
                    'target_score': target_score,
                    'rank': rank,
                    'is_cold': is_cold,
                    'target_novelty': float(self.item_novelty[target_item]),
                })
            
            # Compute metrics for each k
            for k in ks:
                # Compute NDCG
                ndcg_scores = calculate_ndcg(
                    batch_scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'ndcg@{k}'].extend(ndcg_scores.cpu().tolist())
                
                # Compute Hit
                hit_scores = calculate_hit_rate(
                    batch_scores,
                    batch_targets,
                    batch_histories,
                    k=k,
                    use_negatives_only=False
                )
                metrics[f'hit@{k}'].extend(hit_scores.cpu().tolist())
                
                # Compute metrics by Cold/Hot
                for i in range(len(batch_targets)):
                    global_idx = start_idx + i
                    is_cold = global_idx in self.cold_indices
                    
                    if is_cold:
                        metrics[f'cold_ndcg@{k}'].append(ndcg_scores[i].item())
                        metrics[f'cold_hit@{k}'].append(hit_scores[i].item())
                    else:
                        metrics[f'hot_ndcg@{k}'].append(ndcg_scores[i].item())
                        metrics[f'hot_hit@{k}'].append(hit_scores[i].item())
                
                # Compute Novelty metrics
                for i in range(len(batch_targets)):
                    global_idx = start_idx + i
                    target_item = batch_targets[i]

                    # Extract Top-k items
                    top_k_items = torch.topk(batch_scores[i], k).indices.cpu().numpy()

                    # Average Novelty of Top-k items
                    avg_novelty = np.mean([self.item_novelty[item] for item in top_k_items])
                    metrics[f'novelty@{k}'].append(avg_novelty)

                    # Apply target item's Novelty to Hit@k
                    hit_value = hit_scores[i].item()
                    target_novelty = self.item_novelty[target_item]
                    novelty_hit = hit_value * target_novelty
                    metrics[f'novelty_hit@{k}'].append(novelty_hit)
        
        return metrics, rank_info
    
    def evaluate(self, dataset, split="test", save_log=True, pre_generated_texts=None):
        """
        Run the full evaluation pipeline.

        Args:
            dataset: dataset to evaluate
            split: dataset split name ("test", "val", etc.)
            save_log: whether to save log file
            pre_generated_texts: pre-generated texts list (optional)

        Returns:
            results: evaluation results dictionary

        Note:
            - If args.prepend_last_item is True and item_metadata is provided,
              the last purchased item information is prepended to the generated texts.
        """
        print(f"\n{'='*80}")
        print(f"📊 Final Evaluation on {split.upper()} Set")
        print(f"{'='*80}")

        # 1. Collect data
        print("📝 Collecting data...")
        all_prompts = []
        all_targets = []
        all_histories = []
        all_user_ids = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            all_prompts.append(sample["prompt"])
            all_targets.append(sample["target"])
            all_histories.append(sample["history"])
            all_user_ids.append(sample["user_id"])

        if len(all_prompts) > self.args.eval_samples:
            all_prompts = all_prompts[:self.args.eval_samples]
            all_targets = all_targets[:self.args.eval_samples]
            all_histories = all_histories[:self.args.eval_samples]
            all_user_ids = all_user_ids[:self.args.eval_samples]
        
        # 2. Generate text (for all prompts)
        if pre_generated_texts is not None:
            print("📄 Using pre-generated texts from CSV...")
            generated_texts = pre_generated_texts[:len(all_prompts)]
            print(f"  Loaded {len(generated_texts)} pre-generated texts")
        elif hasattr(self.args, "dummy_generation") and self.args.dummy_generation:
            generated_texts = all_prompts
        else:
            generated_texts = self.generate_all_texts(all_prompts)

        # Immediately clean up generation model (save memory)
        if pre_generated_texts is None:  # Only clean up if generation model was used
            print("\n🧹 Cleaning up generation model...")
            if self.llm is not None:
                try:
                    if hasattr(self.llm, 'llm_engine'):
                        del self.llm.llm_engine
                except:
                    pass
                del self.llm
                self.llm = None
            torch.cuda.empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                print(f"💾 GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 2.5. Handle prepend_last_item option
        if hasattr(self.args, 'prepend_last_item') and self.args.prepend_last_item:
            if self.item_metadata is None:
                print("⚠️  Warning: prepend_last_item is enabled but item_metadata is not provided. Skipping prepending.")
            else:
                print("📝 Prepending last purchased item to generated texts...")
                last_item_texts = get_last_item_text(
                    dataset,
                    self.item_metadata,
                    use_brand=getattr(self.args, 'use_brand', True),
                    use_category=getattr(self.args, 'use_category', True)
                )

                # Prepend last item text to generated text
                modified_texts = []
                for last_item_text, generated_text in zip(last_item_texts, generated_texts):
                    if last_item_text:
                        modified_text = f"{last_item_text}\n\n{generated_text}"
                    else:
                        modified_text = generated_text
                    modified_texts.append(modified_text)

                generated_texts = modified_texts
                print(f"✓ Prepended last item to {len(generated_texts)} texts")

                # Print sample
                print("\n" + "="*80)
                print("📝 Sample Modified Text (with last item prepended):")
                print("="*80)
                if len(generated_texts) > 0:
                    print(generated_texts[0][:500] + "..." if len(generated_texts[0]) > 500 else generated_texts[0])
                    print("="*80)
        
        # 3. Compute embeddings (for all generated texts)
        query_embeddings = self.compute_embeddings(generated_texts)

        # Also clean up embedding model (before metric computation)
        print("\n🧹 Cleaning up embedding model...")
        if self.emb_llm is not None:
            try:
                if hasattr(self.emb_llm, 'llm_engine'):
                    del self.emb_llm.llm_engine
            except:
                pass
            del self.emb_llm
            self.emb_llm = None
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            print(f"💾 GPU Memory after embedding: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 4. Compute metrics
        metrics, rank_info = self.compute_retrieval_metrics(
            query_embeddings, 
            all_targets, 
            all_histories,
            ks=[5, 10, 20]
        )
        
        # 5. Save sample outputs (first 3)
        sample_outputs = []
        num_samples = min(3, len(all_prompts))
        for i in range(num_samples):
            sample_outputs.append({
                'prompt': all_prompts[i],
                'generated': generated_texts[i],
                'target': all_targets[i],
                'history': all_histories[i]
            })
        
        # 6. Print results
        self._print_sample_outputs(sample_outputs)
        results = self._print_metrics(metrics, split)
        
        # 7. Save log file
        if save_log:
            self._save_log_file(results, metrics, sample_outputs, split)
            # Save CSV file
            self._save_csv_file(
                all_prompts,
                generated_texts,
                all_targets,
                all_histories,
                all_user_ids,
                rank_info,
                split
            )
            # Save master log file
            self._save_master_log(results, split)
        
        return results
    
    def _print_sample_outputs(self, sample_outputs):
        """Print sample prompts and generated results"""
        print(f"\n{'='*80}")
        print(f"📝 Sample Prompts and Generated Texts")
        print(f"{'='*80}")
        
        for idx, sample in enumerate(sample_outputs, 1):
            print(f"\n[Sample {idx}]")
            print(f"{'─'*80}")
            print(f"Target Item ID: {sample['target']}")
            print(f"History Items: {sample['history']}")
            print(f"\n[Prompt]")
            # Print only first 300 characters if prompt is long
            prompt_preview = sample['prompt'][:300] + "..." if len(sample['prompt']) > 300 else sample['prompt']
            print(prompt_preview)
            print(f"\n[Generated Text]")
            print(sample['generated'])
            print(f"{'─'*80}")
    
    def _print_metrics(self, metrics, split):
        """Print metrics and compute averages"""
        print(f"\n{'='*80}")
        print(f"📈 Final Evaluation Results ({split.upper()})")
        print(f"{'='*80}")

        results = {}

        # Overall metrics
        print("\n[Overall Metrics]")
        for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
        
        # Cold metrics
        print("\n[Cold Items Metrics (80%)]")
        for metric_name in ['cold_hit@5', 'cold_hit@10', 'cold_hit@20', 'cold_ndcg@5', 'cold_ndcg@10', 'cold_ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
            else:
                results[metric_name] = 0.0
                print(f"  {metric_name.upper()}: N/A (no cold items)")
        
        # Hot metrics
        print("\n[Hot Items Metrics (Top 20%)]")
        for metric_name in ['hot_hit@5', 'hot_hit@10', 'hot_hit@20', 'hot_ndcg@5', 'hot_ndcg@10', 'hot_ndcg@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
            else:
                results[metric_name] = 0.0
                print(f"  {metric_name.upper()}: N/A (no hot items)")
        
        # Novelty metrics
        print("\n[Novelty Metrics]")
        for metric_name in ['novelty@5', 'novelty@10', 'novelty@20', 'novelty_hit@5', 'novelty_hit@10', 'novelty_hit@20']:
            if metric_name in metrics and len(metrics[metric_name]) > 0:
                mean_val = np.mean(metrics[metric_name])
                results[metric_name] = float(mean_val)
                print(f"  {metric_name.upper()}: {mean_val:.4f}")
        
        print(f"{'='*80}\n")
        
        return results
    
    def _save_log_file(self, results, metrics, sample_outputs, split):
        """Save evaluation results to a log file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.args.run_name}_{split}_eval.log"
        log_file = results_dir / log_filename
        
        with open(log_file, 'a') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"Evaluation Results - {split.upper()}\n")
            f.write(f"Run Name: {self.args.run_name}\n")
            f.write(f"Dataset: {self.args.data_name}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            # Metric results
            f.write("EVALUATION METRICS\n")
            f.write("-"*80 + "\n")
            
            # Overall Metrics
            f.write("\n[Overall Metrics]\n")
            for metric_name in ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Cold Metrics
            f.write("\n[Cold Items Metrics (80%)]\n")
            for metric_name in ['cold_hit@5', 'cold_hit@10', 'cold_hit@20', 'cold_ndcg@5', 'cold_ndcg@10', 'cold_ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Hot Metrics
            f.write("\n[Hot Items Metrics (Top 20%)]\n")
            for metric_name in ['hot_hit@5', 'hot_hit@10', 'hot_hit@20', 'hot_ndcg@5', 'hot_ndcg@10', 'hot_ndcg@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            # Novelty Metrics
            f.write("\n[Novelty Metrics]\n")
            for metric_name in ['novelty@5', 'novelty@10', 'novelty@20', 'novelty_hit@5', 'novelty_hit@10', 'novelty_hit@20']:
                if metric_name in results:
                    f.write(f"  {metric_name.upper()}: {results[metric_name]:.4f}\n")
            
            f.write("-"*80 + "\n\n")
            
            # Sample outputs
            f.write("SAMPLE PROMPTS AND GENERATED TEXTS\n")
            f.write("="*80 + "\n")
            for idx, sample in enumerate(sample_outputs, 1):
                f.write(f"\n[Sample {idx}]\n")
                f.write("-"*80 + "\n")
                f.write(f"Target Item ID: {sample['target']}\n")
                f.write(f"History Items: {sample['history']}\n")
                f.write(f"\n[Prompt]\n")
                f.write(sample['prompt'] + "\n")
                f.write(f"\n[Generated Text]\n")
                f.write(sample['generated'] + "\n")
                f.write("-"*80 + "\n")
        
        print(f"💾 Evaluation results saved to: {log_file}")
    
    def _save_master_log(self, results, split):
        """
        Record all performance metrics to a single master log file.

        Args:
            results: metrics result dictionary
            split: dataset split name
        """
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        master_log_file = results_dir / f"0_{self.args.data_name}_master.log"

        with open(master_log_file, 'a') as f:
            f.write("="*120 + "\n")
            f.write(f"Master Evaluation Log - {self.args.run_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Dataset: {self.args.data_name}\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write("="*120 + "\n\n")

            # Column headers (Overall)
            f.write("[Overall Metrics]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
                        
            f.write(f"{split}\t"
                   f"{results.get('hit@5', 0.0):.4f}\t"
                   f"{results.get('ndcg@5', 0.0):.4f}\t"
                   f"{results.get('hit@10', 0.0):.4f}\t"
                   f"{results.get('ndcg@10', 0.0):.4f}\t"
                   f"{results.get('hit@20', 0.0):.4f}\t"
                   f"{results.get('ndcg@20', 0.0):.4f}\n")
            
            # Cold Metrics (80%)
            f.write("\n[Cold Items Metrics (80%)]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
        
            f.write(f"{split}\t"
                   f"{results.get('cold_hit@5', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@5', 0.0):.4f}\t"
                   f"{results.get('cold_hit@10', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@10', 0.0):.4f}\t"
                   f"{results.get('cold_hit@20', 0.0):.4f}\t"
                   f"{results.get('cold_ndcg@20', 0.0):.4f}\n")
            
            # Hot Metrics (Top 20%)
            f.write("\n[Hot Items Metrics (Top 20%)]\n")
            f.write("Split\tH@5\tN@5\tH@10\tN@10\tH@20\tN@20\n")
            f.write("-"*70 + "\n")
            
            f.write(f"{split}\t"
                   f"{results.get('hot_hit@5', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@5', 0.0):.4f}\t"
                   f"{results.get('hot_hit@10', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@10', 0.0):.4f}\t"
                   f"{results.get('hot_hit@20', 0.0):.4f}\t"
                   f"{results.get('hot_ndcg@20', 0.0):.4f}\n")
            
            # Novelty Metrics
            f.write("\n[Novelty Metrics]\n")
            f.write("Split\tNov@5\tNH@5\tNov@10\tNH@10\tNov@20\tNH@20\n")
            f.write("-"*70 + "\n")
            
            f.write(f"{split}\t"
                   f"{results.get('novelty@5', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@5', 0.0):.4f}\t"
                   f"{results.get('novelty@10', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@10', 0.0):.4f}\t"
                   f"{results.get('novelty@20', 0.0):.4f}\t"
                   f"{results.get('novelty_hit@20', 0.0):.4f}\n")

            f.write("-"*120 + "\n\n")
        
        print(f"📊 Master log updated: {master_log_file}")
    
    def _save_csv_file(self, prompts, generated_texts, targets, histories, user_ids, rank_info, split):
        """Save evaluation results to a CSV file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.args.run_name}_{split}_eval_{timestamp}.csv"
        csv_file = results_dir / csv_filename

        # Create DataFrame
        data = []
        for i in range(len(prompts)):
            rank = rank_info[i]['rank']
            is_cold = rank_info[i]['is_cold']
            target_novelty = rank_info[i]['target_novelty']
            
            data.append({
                'sample_id': i,
                'user_id': user_ids[i],
                'target_item_id': targets[i],
                'prompt': prompts[i],
                'generated_text': generated_texts[i],
                'history_items': str(histories[i]),  # Convert list to string
                'target_score': rank_info[i]['target_score'],
                'rank': rank,
                'is_cold': is_cold,
                'target_novelty': target_novelty,
                'hit@5': 1 if rank <= 5 else 0,
                'hit@10': 1 if rank <= 10 else 0,
                'hit@20': 1 if rank <= 20 else 0,
                'novelty_hit@5': target_novelty if rank <= 5 else 0,
                'novelty_hit@10': target_novelty if rank <= 10 else 0,
                'novelty_hit@20': target_novelty if rank <= 20 else 0,
            })
        
        df = pd.DataFrame(data)
        
        # Save CSV
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 Detailed results saved to CSV: {csv_file}")
        
        # Print brief statistics
        print(f"\n📈 CSV Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Cold items: {df['is_cold'].sum()}, Warm items: {(~df['is_cold']).sum()}")
        print(f"  Mean rank: {df['rank'].mean():.2f}")
        print(f"  Median rank: {df['rank'].median():.0f}")
        print(f"  Hit@5 rate: {df['hit@5'].mean():.4f}")
        print(f"  Hit@10 rate: {df['hit@10'].mean():.4f}")
        print(f"  Hit@20 rate: {df['hit@20'].mean():.4f}")
        print(f"  Mean target novelty: {df['target_novelty'].mean():.4f}")
    
    def cleanup(self):
        """Clean up memory"""
        print("\n🧹 Cleaning up evaluator resources...")

        # Clean up vLLM generation model
        if self.llm is not None:
            try:
                # Explicitly clean up llm_engine for vLLM
                if hasattr(self.llm, 'llm_engine'):
                    del self.llm.llm_engine
            except:
                pass
            del self.llm
            self.llm = None
            print("  ✓ Generation model cleaned up")
        
        # Clean up embedding model (vLLM or SentenceTransformer)
        if self.emb_llm is not None:
            if not self.use_sentence_transformers:
                # For vLLM
                try:
                    if hasattr(self.emb_llm, 'llm_engine'):
                        del self.emb_llm.llm_engine
                except:
                    pass
            del self.emb_llm
            self.emb_llm = None
            print("  ✓ Embedding model cleaned up")
        
        # Clean up item embeddings
        if self.item_embeddings is not None:
            self.item_embeddings = self.item_embeddings.cpu()
            del self.item_embeddings
            self.item_embeddings = None
            print("  ✓ Item embeddings cleaned up")
        
        # Clean up item popularity and Novelty
        if self.item_popularity is not None:
            del self.item_popularity
            self.item_popularity = None
        
        if self.item_novelty is not None:
            del self.item_novelty
            self.item_novelty = None
        
        if self.hot_items is not None:
            del self.hot_items
            self.hot_items = None
        
        if self.cold_items is not None:
            del self.cold_items
            self.cold_items = None
        
        if any([self.item_popularity is None, self.item_novelty is None, 
                self.hot_items is None, self.cold_items is None]):
            print("  ✓ Item popularity, novelty, and hot/cold sets cleaned up")
        
        # Force release GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  💾 GPU Memory after evaluator cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        print("✓ Evaluator cleanup complete")

