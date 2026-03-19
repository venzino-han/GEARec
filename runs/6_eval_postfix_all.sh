#!/bin/bash

DEVICE=0
SPLIT="train"
MODEL_DIR="gemma-3-1b-it"
EMB_TYPE="item_preference_1024_gemma-3-4b-it"
EMB_MODEL="mixedbread-ai/mxbai-embed-large-v1"
DATASETS=(sports toys yelp)
DATASETS=(beauty sports toys yelp)

# # ─── basic (no postfix) ─────────────────────────────────────────────────────
# for dataset in "${DATASETS[@]}"; do
#     CSV="data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection.csv"
#     [ ! -f "$CSV" ] && echo "⚠️  Skip (not found): $CSV" && continue
#     echo "▶ [basic] ${dataset}"
#     CUDA_VISIBLE_DEVICES=$DEVICE python3 src/eval_key_selection.py \
#         --input_csv "$CSV" \
#         --data_name "$dataset" \
#         --split "$SPLIT" \
#         --emb_type "$EMB_TYPE" \
#         --emb_model_name "$EMB_MODEL" \
#         --use_sentence_transformers \
#         --eval_emb_batch_size 512 \
#         --eval_batch_size 512 \
#         --history_limit 8 \
#         --gpu_memory_utilization 0.5
#     echo "✓ Done: ${dataset} [basic]"
# done

# ─── item_postfix ────────────────────────────────────────────────────────────
for dataset in "${DATASETS[@]}"; do
    # data_processed/beauty_gemma-3-1b-it_train_key_selection_best_from_beauty_gemma-3-1b-it_train_key_selection_suffix_ranked.csv
    CSV="data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection_best_from_${dataset}_${MODEL_DIR}_${SPLIT}_key_selection_suffix_ranked.csv"
    [ ! -f "$CSV" ] && echo "⚠️  Skip (not found): $CSV" && continue
    echo "▶ [postfix] ${dataset}"
    CUDA_VISIBLE_DEVICES=$DEVICE python3 src/eval_key_selection.py \
        --input_csv "$CSV" \
        --data_name "$dataset" \
        --split "$SPLIT" \
        --emb_type "$EMB_TYPE" \
        --emb_model_name "$EMB_MODEL" \
        --use_sentence_transformers \
        --eval_emb_batch_size 512 \
        --eval_batch_size 512 \
        --history_limit 8 \
    # echo "✓ Done: ${dataset} [postfix]"
done

# # ─── preselected ─────────────────────────────────────────────────────────────
# for dataset in "${DATASETS[@]}"; do
#     # preselected CSV may have various name patterns, so search with glob
#     for CSV in data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection_from_*.csv; do
#         [ ! -f "$CSV" ] && continue
#         echo "▶ [preselected] ${dataset}  CSV=${CSV}"
#         CUDA_VISIBLE_DEVICES=$DEVICE python3 src/eval_key_selection.py \
#             --input_csv "$CSV" \
#             --data_name "$dataset" \
#             --split "$SPLIT" \
#             --emb_type "$EMB_TYPE" \
#             --emb_model_name "$EMB_MODEL" \
#             --use_sentence_transformers \
#             --eval_emb_batch_size 512 \
#             --eval_batch_size 512 \
#             --history_limit 8 \
#             --gpu_memory_utilization 0.5
#         echo "✓ Done: ${dataset} [preselected: $CSV]"
#     done
# done

echo ""
echo "All evaluations complete."
ls -lh data_processed/*_ranked.csv 2>/dev/null || echo "(no ranked CSVs found)"
