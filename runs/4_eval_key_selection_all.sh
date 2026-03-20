#!/bin/bash
# =============================================================================
# Stage 2 — Reachability Evaluation (Item-prefix Outputs)
# =============================================================================
# Encodes each (key_item title + query) pair with a dense text encoder and
# computes the retrieval rank of the ground-truth target item against the full
# item pool. Assigns a rank score to every row in the prefix CSV.
# =============================================================================

DEVICE=0
SPLIT="train"
MODEL_DIR="gemma-3-1b-it"
EMB_TYPE="item_preference_1024_gemma-3-4b-it"
EMB_MODEL="mixedbread-ai/mxbai-embed-large-v1"
DATASETS=(beauty sports toys yelp)

# ─── item_prefix ────────────────────────────────────────────────────────────
# Evaluate prefix-mode outputs from Stage 1b
for dataset in "${DATASETS[@]}"; do
    CSV="data_processed/${dataset}_${MODEL_DIR}_${SPLIT}_key_selection_prefix.csv"
    [ ! -f "$CSV" ] && echo "⚠️  Skip (not found): $CSV" && continue
    echo "▶ [prefix] ${dataset}"
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
# Disabled: evaluation of Stage 1c (preselected anchor) outputs
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
