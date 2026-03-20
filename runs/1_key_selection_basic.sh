#!/bin/bash
# =============================================================================
# Preliminary — Zero-shot Key Item Extraction (Basic / Test Split)
# =============================================================================
# Runs generate_key_selection.py on the TEST split for all four datasets
# without any forced anchor. The LLM freely selects the key item from history.
# This is the zero-shot baseline for key item extraction.
# =============================================================================

DEVICE=0
MODEL_NAME="google/gemma-3-1b-it"
SPLIT="test"
DATASETS=(beauty sports toys yelp)

for dataset in "${DATASETS[@]}"; do
    echo "▶ [basic] ${dataset}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name "$dataset" \
        --split "$SPLIT" \
        --model_name "$MODEL_NAME" \
        --history_limit 8 \
        --temperature 0.01 \
        --max_output_tokens 256 \
        --batch_size 32 \
        --gpu_memory_utilization 0.95
    echo "✓ Done: ${dataset}"
done
