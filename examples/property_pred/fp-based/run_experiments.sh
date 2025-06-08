#!/usr/bin/env bash
set -eu

if [ $# -lt 1 ]; then
  echo "Usage: $0 <TASK>"
  echo "  e.g.: $0 AF_APML"
  exit 1
fi

TASK="$1"
# 定义要遍历的参数列表
split_indices=(random1 random2 random3 random4 random5)
split_types=(Random_split Homology_based_split)
model_names=(rf adaboost gradboost knn svm xgboost lightgbm)


# 将项目名设为传入的 TASK，也可以改成固定字符串如 "test_af"
WANDB_PROJECT="$TASK"
fp_type="ecfp"
nbits=2048
radius=3
for st in "${split_types[@]}"; do
    for si in "${split_indices[@]}"; do
        for mn in "${model_names[@]}"; do
        echo "===================================================="
        echo " Running: task=$TASK | model_name=$mn | split_type=$st | split_index=$si"
        echo "===================================================="

        python fp.py --task $TASK --split_type  $st --fp_type $fp_type --nbits $nbits --radius $radius   --split_index $si  --model $mn 
      echo
    done
  done
done
# nohup ./run_experiments.sh AF_APML > AF_APML.log 2>&1 &