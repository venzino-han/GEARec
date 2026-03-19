# GEARec: Grounded Extraction for Anchor-based Recommendation

Official implementation of **GEARec**, a grounded supervision framework for LLM-based sequential recommendation.

GEARec is designed for a practical recommendation setting where a user's interaction history is long, noisy, and only weakly connected to the final target item. Instead of asking an LLM to explain the entire history at once, GEARec first extracts a **high-relevance anchor item** (key item) and then builds **focused reasoning trajectories** around that anchor. This makes the training signal more stable and the final retrieval query more aligned with user behavior.

---

## Abstract

Large Language Models (LLMs) in sequential recommendation often struggle to use long user histories effectively. They tend to overuse irrelevant context and generate reasoning that sounds plausible but does not actually help retrieve the target item. This becomes especially problematic in large candidate pools, where reinforcement learning or free-form exploration rarely produces successful trajectories.

GEARec addresses this issue with two key ideas:

1. **Anchor-Grounded Key Item Extraction**: identify a single key item from user history that most strongly motivates the next purchase, and generate a focused intent paragraph anchored to that item.
2. **Reachability-Aware Trajectory Filtering**: keep only reasoning trajectories whose generated query can actually retrieve the target item within a strict top-K rank threshold.

The resulting dataset is used to fine-tune an LLM. Inference is performed in a **single pass**: the model reads the user history, identifies the key item, generates an anchor-grounded intent query, and retrieves items efficiently via dense embedding search.

---

## Overview

### Why existing methods fail

Existing LLM-based recommendation pipelines often suffer from **context-utilization bias**. The model tries to connect too many past interactions, including noisy or weakly relevant ones, which leads to spurious reasoning and sparse successful trajectories.

<!-- <p align="center">
  <img src="./figs/motiv.png" alt="Existing methods fail due to context-utilization bias and reward sparsity (left), whereas our framework discovers a valid reasoning path guided by a high-relevance anchor (right)." width="700"/>
</p> -->
![Existing methods fail due to context-utilization bias and reward sparsity (left), whereas our framework discovers a valid reasoning path guided by a high-relevance anchor (right).](./figs/motiv.png)


### What GEARec does differently

GEARec identifies a **single decisive key item** from the interaction history, uses that item to generate a focused intent query, and filters out trajectories that are not actually retrievable.

<!-- <p align="center">
  <img src="./figs/method.png" alt="Overview of the proposed framework, GEARec." width="900"/>
</p> -->
![Overview of the proposed framework, GEARec.](./figs/method.png)


---

## Method

GEARec consists of three stages.

### 1. Key Item Extraction

Given a user history, the LLM is prompted to:

- Identify the **single key item** from the history that most strongly motivates the next purchase.
- Generate a detailed, paragraph-length **intent query** (3–5 sentences) explaining why that item influenced the user, what attributes they value, and how this leads to the next purchase.

The output follows a structured tag format:

```
<key_item>[Key Item Name]</key_item>
<query>[A detailed paragraph describing the user's preferences and future intent]</query>
```

To find the best anchor, GEARec runs key item extraction in **item-suffix mode**: for each user, it generates one prompt per history item (appending that item as the forced anchor) and collects N candidate trajectories. This corresponds to `--item_suffix` in `src/generate_key_selection.py`.

---

### 2. Reachability Evaluation

Not every generated query is useful for retrieval. GEARec evaluates each trajectory by:

- Encoding `key_item title + query text` with a dense text encoder (e.g., `mixedbread-ai/mxbai-embed-large-v1`).
- Computing the **retrieval rank** of the ground-truth target item against the full item pool.
- Saving the rank score per trajectory in a ranked CSV.

This step is handled by `src/eval_key_selection.py`.

---

### 3. Best-of-N Sampling and SFT

Using the ranked CSV:

1. **Best-of-N generation** (`src/generate_key_selection.py --ranked_csv`): For each user, the history item with the lowest (best) rank is selected as the anchor. The LLM then generates N samples (e.g., N=8) with temperature > 0 to produce diverse trajectories.
2. **Reachability re-evaluation** (`src/eval_key_selection.py`): The best-of-N samples are ranked again to identify which trajectories can retrieve the target.
3. **SFT on reachable trajectories** (`src/train_sft.py`): Only samples whose rank is within `--max_rank_threshold` (e.g., top-100) are retained for supervised fine-tuning using TRL's `SFTTrainer`.

---

## Running the Pipeline

The pipeline follows four main stages. All scripts are in `runs/`.

### Stage 1 — Key Item Extraction

**1a. Basic (test split, no forced anchor)**

```bash
bash runs/1_key_selection_basic.sh
```

Runs `src/generate_key_selection.py` over the test split for all four datasets.  
Outputs: `data_processed/{dataset}_gemma-3-1b-it_test_key_selection.csv`

**1b. Item-suffix mode (train/valid splits)**

```bash
bash runs/2_key_selection_suffix_titles.sh
```

For each user, generates one prompt per history item (N-item history → N prompts).  
Uses `--item_suffix` flag. Outputs: `data_processed/{dataset}_gemma-3-1b-it_{split}_key_selection_suffix.csv`

**1c. Preselected anchor (test split)**

```bash
bash runs/3_key_selection_preselected.sh
```

Reads an externally provided anchor CSV and injects it via `--preselected_csv`.

---

### Stage 2 — Reachability Evaluation

```bash
bash runs/4_eval_key_selection_all.sh
```

Encodes each `(key_item, query)` pair with `mixedbread-ai/mxbai-embed-large-v1`, computes retrieval rank against the full item pool, and saves the rank column.  
Outputs: `data_processed/{dataset}_gemma-3-1b-it_{split}_key_selection_suffix_ranked.csv`

Key settings:
- `--emb_type item_preference_1024_gemma-3-4b-it`
- `--emb_model_name mixedbread-ai/mxbai-embed-large-v1`

---

### Stage 3 — Best-of-N Sampling and Re-evaluation

**3a. Best-of-N generation**

```bash
bash runs/5_best_postfix_sample8.sh
```

Selects the best anchor per user from the ranked CSV and samples N=8 trajectories at temperature=1.0.  
Outputs: `data_processed/{dataset}_gemma-3-1b-it_{split}_key_selection_best_from_*_ranked.csv`

**3b. Re-evaluate best-of-N**

```bash
bash runs/6_eval_postfix_all.sh
```

Evaluates the best-of-N outputs and assigns retrieval ranks.  
Outputs: `data_processed/{dataset}_..._ranked_ranked.csv`

---

### Stage 4 — SFT Training

```bash
bash runs/7_sft_key_selection.sh
```

Fine-tunes `google/gemma-3-1b-it` using TRL `SFTTrainer` on samples with rank ≤ 100.

Key settings:

| Parameter | Value |
|---|---|
| `--max_rank_threshold` | 100 |
| `--max_steps` | 3000 |
| `--learning_rate` | 1e-7 |
| `--batch_size` | 2 |
| `--prompt_type` | `key_selection` |
| `--target_type` | `key_selection` |
| `--emb_model_name` | `mixedbread-ai/mxbai-embed-large-v1` |

Checkpoints are saved to `checkpoints/sft_keysel_{dataset}_seed{seed}_rank{threshold}_steps{max_steps}/`.

---

## Output Format

The LLM is trained to produce structured output in the following format:

```
<key_item>Wireless Earbuds</key_item>
<query>
The user has consistently gravitated toward portable audio accessories, ...
This suggests a strong intent to explore complementary wireless audio products ...
</query>
```

The `<query>` text is encoded for retrieval at inference time.

---

## Key Design Choices

- **Anchor before reasoning**: prevents the LLM from over-explaining the full user history.
- **Item-suffix forcing**: systematically tests every history item as a potential anchor to find the most retrievable one.
- **Reachability filtering**: keeps only reasoning that leads to actual retrieval success (rank ≤ threshold).
- **Best-of-N at high temperature**: explores diverse phrasings anchored to the best item before SFT filtering.
- **Single-pass inference**: at test time, only one forward pass through the LLM is needed.

---

## Expected Outcome

GEARec is intended for researchers who want to study how LLM reasoning can be made more grounded, retrieval-aware, and practically useful in recommendation systems. The main hypothesis is that recommendation quality improves when supervision is built from **reachable anchor-grounded reasoning trajectories** rather than unconstrained free-form explanations.

---

## Citation

```bibtex
@article{anonymous2026gearec,
  title   = {Illuminating the Path: Grounded Extraction for Anchor-based Recommendation},
  author  = {Anonymous ACL Submission},
  year    = {2026}
}
```
