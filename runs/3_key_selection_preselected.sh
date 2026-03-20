#!/bin/bash
# =============================================================================
# Stage 2 — Key Item Extraction (Preselected Anchor)
# =============================================================================
# Reads a preselected anchor CSV (typically the output of Stage 1a) and
# re-runs generation with that fixed anchor injected via --preselected_csv.
# Produces exactly 1 prompt per user (anchor is already determined).
# Useful for ablations or re-generating queries from a known anchor set.
# =============================================================================

DEVICE=0
MODEL_NAME="google/gemma-3-1b-it"
SPLIT="test"
MODEL_DIR="gemma-3-1b-it"
DATASETS=(beauty sports toys yelp)

for dataset in "${DATASETS[@]}"; do
    CSV="data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection.csv"
    if [ ! -f "$CSV" ]; then
        echo "⚠️  Skipping ${dataset}: CSV not found (${CSV})"
        continue
    fi
    echo "▶ [preselected] ${dataset}  CSV=${CSV}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name "$dataset" \
        --split "$SPLIT" \
        --model_name "$MODEL_NAME" \
        --preselected_csv "$CSV"    # inject pre-determined anchor from the CSV
    echo "✓ Done: ${dataset}"
done
