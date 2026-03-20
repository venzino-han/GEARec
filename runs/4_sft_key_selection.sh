#!/bin/bash
# SFT Training with CF-augmented key-selection data
#
# 동작 순서 (train_sft.py 내부에서 수행):
#   1. main_csv 에서 rank > MAIN_RANK_THRESHOLD 인 "미커버 유저" 선별
#   2. cf_csv   에서 rank <= CF_RANK_THRESHOLD  인 케이스 중 미커버 유저만 선택하여
#      generated_text 치환
#   3. 합산된(main 커버 + cf 보강) 데이터로 SFT 학습 수행
#
# cf_csv 파일이 없는 데이터셋은 main_csv 단독으로 학습 (경고 출력 후 계속).

# ─── 모델 설정 ────────────────────────────────────────────────────────────────
MODEL_NAME="google/gemma-3-1b-it"
MODEL_DIR="gemma-3-1b-it"

# ─── rank 임계값 ──────────────────────────────────────────────────────────────
MAIN_RANK_THRESHOLD=20   # main_csv 에서 rank <= 이 값: 이미 잘 학습된 케이스
GT_RANK_THRESHOLD=100     # gt_based_csv   에서 rank <= 이 값: 치환에 사용할 CF 케이스

# ─── 학습 설정 ────────────────────────────────────────────────────────────────
BATCH_SIZE=4
EVAL_BATCH_SIZE=32
EPOCHS=1
DAYS_FILTER=365
LEARNING_RATE=1e-6
MAX_HISTORY_LEN=8
MAX_STEPS=3000
SAVE_STEPS=$MAX_STEPS
PROMPT_TYPE="key_selection"
LOSS_TYPES=("dft")
GRADIENT_ACCUMULATION_STEPS=1

# ─── 데이터셋 ─────────────────────────────────────────────────────────────────
DATASETS=(beauty toys sports yelp)

# ─── main key-selection CSV (기존 best_target_text) ───────────────────────────
declare -A MAIN_CSV
MAIN_CSV[beauty]="data_processed/beauty_gemma-3-1b-it_train_key_selection_best_target_text.csv"
MAIN_CSV[toys]="data_processed/toys_gemma-3-1b-it_train_key_selection_best_target_text.csv"
MAIN_CSV[sports]="data_processed/sports_gemma-3-1b-it_train_key_selection_best_target_text.csv"
MAIN_CSV[yelp]="data_processed/yelp_gemma-3-1b-it_train_key_selection_best_target_text.csv"

# ─── CF key-selection CSV (보강용 cf_best_target_text) ────────────────────────
# 아직 없으면 해당 데이터셋은 main_csv 단독으로 학습.
declare -A CF_CSV
CF_CSV[beauty]="data_processed/beauty_gemma-3-1b-it_train_key_selection_gt_based_ranked.csv"
CF_CSV[toys]="data_processed/toys_gemma-3-1b-it_train_key_selection_gt_based_ranked.csv"
CF_CSV[sports]="data_processed/sports_gemma-3-1b-it_train_key_selection_gt_based_ranked.csv"
CF_CSV[yelp]="data_processed/yelp_gemma-3-1b-it_train_key_selection_gt_based_ranked.csv"

# ─── 장치 설정 ────────────────────────────────────────────────────────────────
TRACKER="python3 utils/device_tracker.py"
device=0

# ─────────────────────────────────────────────────────────────────────────────

for seed in 42; do
for data_name in "${DATASETS[@]}"; do
for loss_type in "${LOSS_TYPES[@]}"; do
    MAIN="${MAIN_CSV[$data_name]}"
    CF="${CF_CSV[$data_name]}"

    # main CSV 존재 확인
    if [ -z "$MAIN" ] || [ ! -f "$MAIN" ]; then
        echo "⚠️  Skip ${data_name}: main CSV not found → $MAIN"
        continue
    fi

    RUN_NAME="sft_keysel_gt_guided_${data_name}_seed${seed}_main${MAIN_RANK_THRESHOLD}_gt${GT_RANK_THRESHOLD}_days${DAYS_FILTER}_steps${MAX_STEPS}_${loss_type}"
    CHECKPOINT_DIR="checkpoints/${RUN_NAME}"
    FINAL_CHECKPOINT_DIR="${CHECKPOINT_DIR}/checkpoint-${MAX_STEPS}"

    $TRACKER allocate $device "$RUN_NAME"

    BRAND_FLAG="--use_brand"
    if [ $data_name == yelp ]; then
        BRAND_FLAG=""
    fi

    # CF CSV 존재 여부에 따라 --cf_key_selection_csv 플래그 결정
    CF_FLAG=""
    if [ -n "$CF" ] && [ -f "$CF" ]; then
        CF_FLAG="--cf_key_selection_csv $CF --cf_rank_threshold $GT_RANK_THRESHOLD"
    else
        echo "ℹ️  [${data_name}] CF CSV not found (${CF}). main_csv 단독 학습."
    fi

    echo ""
    echo "================================================="
    echo "  SFT Training (CF-augmented key_selection)"
    echo "================================================="
    echo "  Dataset        : $data_name"
    echo "  Seed           : $seed"
    echo "  main CSV       : $MAIN"
    echo "  cf   CSV       : ${CF:-N/A}"
    echo "  main rank thr  : $MAIN_RANK_THRESHOLD"
    echo "  cf   rank thr  : $GT_RANK_THRESHOLD"
    echo "================================================="
    echo ""

    CUDA_VISIBLE_DEVICES=$device python3 src/train_sft.py \
        --data_name            $data_name \
        --model_name           $MODEL_NAME \
        --seed                 $seed \
        --run_name             $RUN_NAME \
        --prompt_type          $PROMPT_TYPE \
        --loss_type            $loss_type \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        $BRAND_FLAG \
        --use_category \
        --emphasize_recent_item \
        --target_type          key_selection \
        --key_selection_csv    $MAIN \
        --max_rank_threshold   $MAIN_RANK_THRESHOLD \
        $CF_FLAG \
        --emb_model_name       "mixedbread-ai/mxbai-embed-large-v1" \
        --emb_type             item_preference_1024_gemma-3-4b-it \
        --max_steps            $MAX_STEPS \
        --save_steps           $MAX_STEPS \
        --checkpoint_dir       $CHECKPOINT_DIR \
        --final_checkpoint_dir $FINAL_CHECKPOINT_DIR \
        --num_epochs           $EPOCHS \
        --batch_size           $BATCH_SIZE \
        --eval_batch_size      $EVAL_BATCH_SIZE \
        --learning_rate        $LEARNING_RATE \
        --max_history_len      $MAX_HISTORY_LEN \
        --days_filter          $DAYS_FILTER \
        --run_evaluation \
        "$@"

    $TRACKER free $device
done
done
done
