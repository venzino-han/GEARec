#!/bin/bash
# SFT Training using key-selection ranked CSV
# eval_key_selection.py 로 생성된 ranked CSV에서
# rank <= MAX_RANK_THRESHOLD 인 샘플만 학습에 사용

# ─── 모델 설정 ────────────────────────────────────────────────────────────────
MODEL_NAME="google/gemma-3-1b-it"
MODEL_DIR="gemma-3-1b-it"

# ─── 필터 임계값 ──────────────────────────────────────────────────────────────
MAX_RANK_THRESHOLD=100   # rank <= 이 값인 샘플만 학습에 사용

# ─── 학습 설정 ────────────────────────────────────────────────────────────────
BATCH_SIZE=2
EVAL_BATCH_SIZE=16
EPOCHS=1
LEARNING_RATE=1e-7
MAX_HISTORY_LEN=8
MAX_STEPS=3000
SAVE_STEPS=1000
PROMPT_TYPE="key_selection"

# ─── 데이터셋 ─────────────────────────────────────────────────────────────────
DATASETS=(beauty toys sports yelp)

# ─── key-selection ranked CSV (데이터셋별 명시적 경로) ───────────────────────
declare -A KEY_SELECTION_CSV
KEY_SELECTION_CSV[beauty]="data_processed/beauty_gemma-3-1b-it_train_key_selection_best_from_beauty_gemma-3-1b-it_train_key_selection_suffix_ranked_ranked.csv"
KEY_SELECTION_CSV[toys]="data_processed/toys_gemma-3-1b-it_train_key_selection_best_from_toys_gemma-3-1b-it_train_key_selection_suffix_ranked_ranked.csv"
KEY_SELECTION_CSV[sports]="data_processed/sports_gemma-3-1b-it_train_key_selection_best_from_sports_gemma-3-1b-it_train_key_selection_suffix_ranked_ranked.csv"
KEY_SELECTION_CSV[yelp]="data_processed/yelp_gemma-3-1b-it_train_key_selection_best_from_yelp_gemma-3-1b-it_train_key_selection_suffix_ranked_ranked.csv"

# ─── 장치 설정 ────────────────────────────────────────────────────────────────
TRACKER="python3 utils/device_tracker.py"
device=2

# ─────────────────────────────────────────────────────────────────────────────

for seed in 42; do
for data_name in "${DATASETS[@]}"; do

    CSV="${KEY_SELECTION_CSV[$data_name]}"

    if [ -z "$CSV" ]; then
        echo "⚠️  Skip ${data_name}: KEY_SELECTION_CSV 미정의"
        continue
    fi
    if [ ! -f "$CSV" ]; then
        echo "⚠️  Skip ${data_name}: CSV not found → $CSV"
        continue
    fi

    RUN_NAME="sft_keysel_${data_name}_seed${seed}_rank${MAX_RANK_THRESHOLD}_steps${MAX_STEPS}"
    CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
    FINAL_CHECKPOINT_DIR="${CHECKPOINT_DIR}/checkpoint-${MAX_STEPS}"

    $TRACKER allocate $device "$RUN_NAME"

    echo ""
    echo "================================="
    echo "  SFT Training (key_selection)"
    echo "================================="
    echo "  Dataset  : $data_name"
    echo "  Seed     : $seed"
    echo "  CSV      : $CSV"
    echo "  Max rank : $MAX_RANK_THRESHOLD"
    echo "================================="
    echo ""

    CUDA_VISIBLE_DEVICES=$device python3 src/train_sft.py \
        --data_name            $data_name \
        --model_name           $MODEL_NAME \
        --seed                 $seed \
        --run_name             $RUN_NAME \
        --prompt_type          $PROMPT_TYPE \
        --use_brand \
        --use_category \
        --emphasize_recent_item \
        --target_type          key_selection \
        --key_selection_csv    $CSV \
        --max_rank_threshold   $MAX_RANK_THRESHOLD \
        --emb_model_name       "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type             item_preference_1024_gemma-3-4b-it \
        --max_steps            $MAX_STEPS \
        --save_steps           $SAVE_STEPS \
        --checkpoint_dir       $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --num_epochs           $EPOCHS \
        --batch_size           $BATCH_SIZE \
        --eval_batch_size      $EVAL_BATCH_SIZE \
        --learning_rate        $LEARNING_RATE \
        --max_history_len      $MAX_HISTORY_LEN \
        --max_new_tokens       32 \
        --gpu_memory_utilization 0.95 \
        --run_evaluation \
        "$@"

    $TRACKER free $device

    echo ""
    echo "✅ Completed: $RUN_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

done
done

echo ""
echo "🎉 All SFT (key_selection) training jobs completed!"
