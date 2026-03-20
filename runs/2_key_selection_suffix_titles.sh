#!/bin/bash
# =============================================================================
# Stage 1 — Key Item Extraction (Item-prefix Mode / Train + Valid Splits)
# =============================================================================
# For each user, appends every history item title as a forced postfix anchor
# to the prompt, producing N prompts per user (one per history item).
# This systematically tests each candidate anchor to discover which one yields
# the most retrievable intent query.
# =============================================================================

DEVICE=0
MODEL_NAME="google/gemma-3-1b-it"
SPLITS=(train valid test)
DATASETS=(beauty sports toys yelp)

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
    echo "▶ [item_prefix] ${dataset}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name "$dataset" \
        --split "$split" \
        --model_name "$MODEL_NAME" \
        --item_prefix
    echo "✓ Done: ${dataset} ${split}"
    done
done
