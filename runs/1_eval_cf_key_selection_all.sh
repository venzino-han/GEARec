#!/bin/bash

# Generaate and for CF prefix + GT generation for train dataset 

DEVICE=0
MODEL_DIR="gemma-3-1b-it"
EMB_TYPE="item_preference_1024_gemma-3-4b-it"
EMB_MODEL="mixedbread-ai/mxbai-embed-large-v1"
DATASETS=(beauty sports toys)
SPLITS=(train valid)


for dataset in "${DATASETS[@]}"; do
    for split in "${SPLITS[@]}"; do
    CSV="data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection_retrieval_rank_top100_from_SASRec_${dataset}_${SPLIT}_trigger_items.csv"
    [ ! -f "$CSV" ] && echo "⚠️  Skip (not found): $CSV" && continue
    echo "▶ [CF prefix] ${dataset}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/eval_key_selection.py \
        --input_csv "$CSV" \
        --data_name "$dataset" \
        --split "$SPLIT" \
        --emb_type "$EMB_TYPE" \
        --emb_model_name "$EMB_MODEL" \
        --use_sentence_transformers \
        --eval_emb_batch_size 512 \
        --eval_batch_size 512
    done
done
