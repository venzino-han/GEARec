#!/bin/bash

EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1"
CUDA_DEV=0
EMB_PREFIX="sft_keysel_main20_cf_guided_user_emb"

# data_name:csv_file:split"
EXTRACT_TASKS=(
    "beauty:results/sft_keysel_cf_guided_beauty_seed42_main20_cf100_days365_steps3000_dft_train_eval_20260307_155024.csv:train"
    "beauty:results/sft_keysel_cf_guided_beauty_seed42_main20_cf100_days365_steps3000_dft_valid_eval_20260307_160740.csv:val"
    "beauty:results/sft_keysel_cf_guided_beauty_seed42_main20_cf100_days365_steps3000_dft_test_eval_20260306_215209.csv:test"

    "sports:results/sft_keysel_cf_guided_sports_seed42_main20_cf100_days365_steps3000_dft_train_eval_20260307_154602.csv:train"
    "sports:results/sft_keysel_cf_guided_sports_seed42_main20_cf100_days365_steps3000_dft_valid_eval_20260307_161253.csv:val"
    "sports:results/sft_keysel_cf_guided_sports_seed42_main20_cf100_days365_steps3000_dft_test_eval_20260306_233529.csv:test"

    "toys:results/sft_keysel_cf_guided_toys_seed42_main20_cf100_days365_steps3000_dft_train_eval_20260307_155323.csv:train"
    "toys:results/sft_keysel_cf_guided_toys_seed42_main20_cf100_days365_steps3000_dft_valid_eval_20260307_161522.csv:val"
    "toys:results/sft_keysel_cf_guided_toys_seed42_main20_cf100_days365_steps3000_dft_test_eval_20260306_223619.csv:test"

    "yelp:results/sft_keysel_cf_guided_yelp_seed42_main20_cf100_days365_steps3000_dft_train_eval_20260307_163919.csv:train"
    "yelp:results/sft_keysel_cf_guided_yelp_seed42_main20_cf100_days365_steps3000_dft_valid_eval_20260307_170007.csv:val"
    "yelp:results/sft_keysel_cf_guided_yelp_seed42_main20_cf100_days365_steps3000_dft_test_eval_20260307_172050.csv:test"
)

for task in "${EXTRACT_TASKS[@]}"; do
    IFS=':' read -r data_name csv_file split <<< "$task"
    output_emb="data_emb/${EMB_PREFIX}_${data_name}_${split}.pt"

    CUDA_VISIBLE_DEVICES=$CUDA_DEV python3 src/extract_user_emb.py \
        --run_name "${data_name}_sft_keysel_cf_guided_prepend" \
        --data_name "$data_name" \
        --batch_size 512 \
        --target_file "$csv_file" \
        --embedding_model_name "$EMBEDDING_MODEL" \
        --output_file "$output_emb" \
        --use_sentence_transformers \
        --prepend_last_item_meta \
        --split "$split"
done
