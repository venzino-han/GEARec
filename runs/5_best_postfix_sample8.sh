#!/bin/bash
# Automatically select the postfix_item_id with the lowest rank per user from the ranked CSV
# (selection logic is handled internally by generate_key_selection.py),
# generate 8 samples with temperature=1.0

DEVICE=0
MODEL_NAME="google/gemma-3-1b-it"
MODEL_DIR="gemma-3-1b-it"
SPLITS=(train valid)
DATASETS=(beauty sports toys yelp)
N_SAMPLES=8
TEMPERATURE=1.0

for dataset in "${DATASETS[@]}"; do
for split in "${SPLITS[@]}"; do

    RANKED_CSV="data_processed/${dataset}_${MODEL_DIR}_${split}_key_selection_suffix_ranked.csv"
    if [ ! -f "$RANKED_CSV" ]; then
        echo "⚠️  Skip (not found): $RANKED_CSV"
        continue
    fi

    echo "▶ [best_postfix x${N_SAMPLES} samples] ${dataset} / ${split}"

    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/generate_key_selection.py \
        --data_name   "$dataset" \
        --split       "$split" \
        --model_name  "$MODEL_NAME" \
        --history_limit 8 \
        --word_limit  256 \
        --days        365 \
        --temperature $TEMPERATURE \
        --max_output_tokens 512 \
        --batch_size  32 \
        --gpu_memory_utilization 0.95 \
        --ranked_csv  "$RANKED_CSV" \
        --n_samples   $N_SAMPLES

    echo "✓ Done: ${dataset} / ${split}  (${N_SAMPLES} samples)"
    echo "──────────────────────────────────────────────────────"

done
done

echo ""
echo "All datasets and splits complete."
echo ""
echo "Generated files:"
ls -lh data_processed/*_best_from_*.csv 2>/dev/null \
    || echo "  (no output files found)"
