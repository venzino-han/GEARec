#!/bin/bash

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
        --word_limit 128 \
        --days 365 \
        --temperature 0.01 \
        --max_output_tokens 256 \
        --batch_size 16 \
        --gpu_memory_utilization 0.95
    echo "✓ Done: ${dataset}"
done
