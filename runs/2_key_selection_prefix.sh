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
SPLITS=(train valid)
DATASETS=(beauty sports toys)

EMB_TYPE="item_preference_1024_gemma-3-4b-it"
EMB_MODEL="mixedbread-ai/mxbai-embed-large-v1"
MODEL_DIR="gemma-3-1b-it"

for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
    echo "▶ [prefix] ${dataset} ${split}"

    # Generate reasoning with item-prefix mode
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name "$dataset" \
        --split "$split" \
        --model_name "$MODEL_NAME" \
        --item_prefix
    echo "✓ Done: ${dataset} ${split}"

    # Evaluate reasoning with item-prefix mode
    CSV="data_processed/${dataset}_${MODEL_DIR}_${split}_key_selection_prefix.csv"
    [ ! -f "$CSV" ] && echo "⚠️  Skip (not found): $CSV" && continue
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/eval_key_selection.py \
        --input_csv "$CSV" \
        --data_name "$dataset" \
        --split "$split" \
        --emb_type "$EMB_TYPE" \
        --emb_model_name "$EMB_MODEL" \
        --use_sentence_transformers \
        --eval_emb_batch_size 512 \
        --eval_batch_size 512
    done
done

