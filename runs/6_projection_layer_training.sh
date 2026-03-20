#!/bin/bash

CUDA_DEV=0
EMB_PREFIX="sft_keysel_main20_cf_guided_user_emb"
DATASET_CONFIGS=(
    "beauty:train:val:test:0"
    "toys:train:val:test:0"
    "sports:train:val:test:0"
    "yelp:train:train:test:0"
)

for config in "${DATASET_CONFIGS[@]}"; do
    IFS=':' read -r data_name train_split val_split test_split copy_val <<< "$config"

    train_emb="data_emb/${EMB_PREFIX}_${data_name}_${train_split}.pt"
    val_emb="data_emb/${EMB_PREFIX}_${data_name}_${val_split}.pt"
    test_emb="data_emb/${EMB_PREFIX}_${data_name}_${test_split}.pt"

    if [[ "$copy_val" == "1" ]]; then
        cp "$train_emb" "$val_emb"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python3 src/train_embedding.py \
        --data_name "$data_name" \
        --train_user_embedding_file "$train_emb" \
        --val_user_embedding_file   "$val_emb" \
        --test_user_embedding_file  "$test_emb" \
        --epochs 10 \
        --learning_rate 1e-4 \
        --embedding_dim 1024 \
        --train_batch_size 128 \
        --eval_batch_size 512 \
        --beta 0.001 \
        --dropout 0.2 \
        --seed 42 \
        --item_embedding_file "./data_emb/${data_name}_item_preference_1024_gemma-3-4b-it_mxbai-embed-large-v1_emb.pt" \
        --run_name "${data_name}_sft_keysel_cf_guided_projection"
done
        # --zero_shot \
        # --use_crm \
