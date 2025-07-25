#!/usr/bin/env bash
set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 <TASK>"
  echo "  e.g.: $0 AF_APML"
  exit 1
fi

TASK="$1"
# 定义要遍历的参数列表
fold_seeds=(0 1 2 3 4)
split_types=(random_split mmseqs2_split cdhit_split)
model_names=(rf adaboost gradboost knn xgboost lightgbm)


# 将项目名设为传入的 TASK，也可以改成固定字符串如 "test_af"
WANDB_PROJECT="$TASK"
fp_type="ecfp6"
for st in "${split_types[@]}"; do
    for si in "${fold_seeds[@]}"; do
        for mn in "${model_names[@]}"; do
        echo "===================================================="
        echo " Running: task=$TASK | model_name=$mn | split_type=$st | fold_seed=$si"
        echo "===================================================="

        python fp.py --task $TASK --split_type  $st --fp_type $fp_type  --fold_seed $si  --model $mn 
      echo
    done
  done
done