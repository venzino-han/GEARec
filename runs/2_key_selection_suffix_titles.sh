#!/bin/bash
# Generate by appending history item titles as postfix after applying chat template
# N history items → N prompts per user

DEVICE=0
MODEL_NAME="google/gemma-3-1b-it"
SPLITS=(train valid)
DATASETS=(beauty sports toys yelp)

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
    echo "▶ [item_postfix] ${dataset}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name "$dataset" \
        --split "$split" \
        --model_name "$MODEL_NAME" \
        --history_limit 8 \
        --word_limit 256 \
        --days 365 \
        --temperature 0.01 \
        --max_output_tokens 512 \
        --batch_size 32 \
        --gpu_memory_utilization 0.95 \
        --item_suffix
    echo "✓ Done: ${dataset} ${split}"
    done
done
