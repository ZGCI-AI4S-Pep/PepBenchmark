#!/usr/bin/env bash
set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 <TASK>"
  echo "  e.g.: $0 AF_APML"
  exit 1
fi

TASK="$1"

# 定义要遍历的参数列表
split_indices=(0 1 2 3 4)
<<<<<<< HEAD
split_types=(random_split mmseqs2_split)
=======
split_types=(random_split mmseqs2_split cdhit_split)
>>>>>>> origin/main
model_names=(facebook/esm2_t30_150M_UR50D)

# 可选：设置公共超参
NUM_EPOCHS=30
BATCH_SIZE=64
ACCUM_STEPS=1
LR=5e-5
PATIENCE=5
WEIGHT_DECAY=0.0

# 将项目名设为传入的 TASK，也可以改成固定字符串如 "test_af"
WANDB_PROJECT="$TASK"



for st in "${split_types[@]}"; do
    for si in "${split_indices[@]}"; do
        for mn in "${model_names[@]}"; do


        echo "===================================================="
        echo " Running: task=$TASK | model_name=$mn | split_type=$st | fold_seed=$si"
        echo "===================================================="


        WANDB_PROJECT=$WANDB_PROJECT python finetune_plm.py \
            --num_train_epochs $NUM_EPOCHS \
            --load_best_model_at_end \
            --task $TASK \
            --per_device_train_batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $ACCUM_STEPS \
            --learning_rate $LR \
            --early_stopping_patience $PATIENCE \
            --weight_decay $WEIGHT_DECAY \
            --split_type $st \
            --fold_seed $si \
            --model_name $mn

      echo
    done
  done
done
# nohup ./run_experiments.sh BPP_APML > BPP_APML.log 2>&1 &