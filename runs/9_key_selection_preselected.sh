#!/bin/bash
# 외부 CSV의 selected_item_id를 챗 템플릿 이후 postfix로 붙여 생성
# 유저당 1개 프롬프트

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
        --history_limit 8 \
        --word_limit 128 \
        --days 365 \
        --temperature 0.01 \
        --max_output_tokens 256 \
        --batch_size 16 \
        --gpu_memory_utilization 0.95 \
        --preselected_csv "$CSV"
    echo "✓ Done: ${dataset}"
done
