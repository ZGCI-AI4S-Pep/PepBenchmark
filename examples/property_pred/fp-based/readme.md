# 基于分子指纹的肽段性质预测

本目录包含基于分子指纹进行肽段性质预测的代码。

## 环境要求
使用 `pepbenchmark` 环境即可。

## 主要文件

### 1. fp.py - 模型训练和评估（推荐）
用于训练模型和评估性能的主文件，**推荐使用默认参数，调参后性能并没有提升**。

#### 使用方法：
```bash
python fp.py --task Antimicrobial --model lightgbm --split_type random_split --fp_type ecfp6 --output_dir ./checkpoints
```
#### 主要参数：
- `--task`: 任务名称（如 AV_APML）
- `--model`: 模型类型（rf, adaboost, gradboost, knn, svm, xgboost, lightgbm），默认 rf
- `--split_type`: 数据分割类型（random_split, mmseqs2_split），默认 mmseqs2_split
- `--fp_type`: 指纹类型（如 ecfp6），默认 ecfp6
- `--fold_seed`: 交叉验证种子（0-4），默认 0
- `--output_dir`: 输出目录，默认 ./checkpoints

#### 工作流程：
1. 自动查找 `best_params_{model}.json` 文件 （`{output_dir}/{task}/{split_type}/{model}/{fold_seed}/`下）
2. 如果找到，使用最佳参数；否则使用默认参数（推荐）
3. 训练模型并在训练集、验证集、测试集上评估
4. 保存训练好的模型和评估指标

#### 输出：
结果保存到 `{output_dir}/{task}/{split_type}/{model}/{fold_seed}/`，包含：
- `model.joblib`: 训练好的模型
- `metrics.csv`: 评估指标结果

> Model,Fingerprint,accuracy,balanced-accuracy,precision,recall,specificity,f1,micro-f1,macro-f1,weighted-f1,mcc,kappa,g-mean,roc-auc,avg-roc-auc,pr-auc,brier-score,log-loss
rf,ecfp6,0.9932221063607924,0.9919465758548958,0.9910562180579217,0.9978559176672385,0.9860372340425532,0.9944444444444445,0.9932221063607924,0.9928773024361259,0.9932155697935003,0.985792743596201,0.9857548653070926,0.9919289737826985,0.9997010544779387,0.9997010544779387,0.999799900536787,0.057076813241628485,0.24481178078052915
rf,ecfp6,0.826722338204593,0.7938527272027407,0.844311377245509,0.9009584664536742,0.6867469879518072,0.8717156105100463,0.826722338204593,0.8024172907855698,0.8236842072772148,0.608735598279715,0.6056987573019667,0.7865942493476163,0.9009007275106817,0.9009007275106817,0.9480090828754691,0.13795231296927135,0.4302468343766707
rf,ecfp6,0.7920997920997921,0.7605163751054389,0.7964071856287425,0.8926174496644296,0.6284153005464481,0.8417721518987342,0.7920997920997921,0.7693709244342156,0.7866809892126349,0.5491194140911866,0.5415904239097287,0.7489555813957705,0.8840539846701141,0.8840539846701141,0.9243416992945246,0.14905599009867357,0.46037712238272643



### 2. fp_tune.py - 超参数调优（可选）
专门用于使用Optuna进行超参数调优的文件。**注意：默认参数通常已经很好，大多数情况下不需要调优**。

#### 使用方法：
```bash
python fp_tune.py --task AV_APML --split_type random_split --fp_type ecfp6 --model rf --n_trials 10
```


#### 输出：
- `best_params_{model}.json`: 最佳超参数
- `tuning_trials_{model}.csv`: 所有调优试验结果
- `model.joblib`: 使用最佳参数训练的最终模型
- `metrics.csv`: 最终模型的评估指标结果

### 3. run_one_dataset.sh - 批量运行脚本
一次性跑一个数据集的两种划分方式的5个种子：
```bash
bash run_one_dataset.sh Antimicrobial
```



### 4. run_all.sh 
调用run_one_dataset.sh ,跑所有数据




