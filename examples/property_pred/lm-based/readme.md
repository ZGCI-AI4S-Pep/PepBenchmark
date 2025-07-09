# 基于预训练语言模型的肽段性质预测

本目录包含基于预训练蛋白质语言模型进行肽段性质预测的代码。

## 环境要求
使用 `pepbenchmark` 环境即可。

## 主要文件

### 1. finetune_plm.py - 模型训练和评估（推荐）
用于训练模型和评估性能的主文件，**推荐使用默认参数，调参后性能并没有显著提升**。

#### 使用方法：
```bash
python finetune_plm.py --task AV_APML --split_type random_split --model_name facebook/esm2_t30_150M_UR50D --fold_seed 0 --output_dir ./checkpoints --load_best_model_at_end
```

#### 主要参数：
- `--task`: 任务名称（如 AV_APML）
- `--model_name`: 预训练模型名称，默认 facebook/esm2_t30_150M_UR50D
- `--split_type`: 数据分割类型（random_split, mmseqs2_split），默认 random_split
- `--fold_seed`: 交叉验证种子（0-4），默认 0
- `--num_train_epochs`: 训练轮数，默认 30
- `--learning_rate`: 学习率，默认 5e-5
- `--per_device_train_batch_size`: 批次大小，默认 64
- `--early_stopping_patience`: 早停耐心值，默认 5
- `--output_dir`: 输出目录，默认 ./checkpoints
- `--load_best_model_at_end`: 是否加载最好的模型

#### 工作流程：
1. 加载预训练的蛋白质语言模型（如 ESM2）
2. 在训练集上微调模型
3. 在训练集、验证集、测试集上评估性能
4. 保存训练好的模型和评估指标

#### 输出：
结果保存到 `{output_dir}/{task}/{split_type}/{model_name}/{fold_seed}/`，包含：
- `model/`: 训练好的模型文件
- `tokenizer/`: 分词器文件
- `metrics.csv`: 评估指标结果
- `training_config.json`: 训练配置

> Split,Model,accuracy,balanced-accuracy,precision,recall,specificity,f1,micro-f1,macro-f1,weighted-f1,mcc,kappa,g-mean,roc-auc,avg-roc-auc,pr-auc,brier-score,log-loss
Train,facebook/esm2_t30_150M_UR50D,0.995,0.994,0.996,0.994,0.995,0.995,0.995,0.995,0.995,0.989,0.989,0.994,0.999,0.999,0.999,0.010,0.020
Validation,facebook/esm2_t30_150M_UR50D,0.872,0.846,0.889,0.901,0.791,0.895,0.872,0.846,0.870,0.698,0.695,0.845,0.951,0.951,0.978,0.098,0.321
Test,facebook/esm2_t30_150M_UR50D,0.834,0.812,0.851,0.892,0.732,0.871,0.834,0.812,0.832,0.635,0.631,0.810,0.924,0.924,0.967,0.125,0.398

### 2. finetune_plm_optuna.py - 超参数调优（可选）
专门用于使用Optuna进行超参数调优的文件。**注意：默认参数通常已经很好，大多数情况下不需要调优**。

#### 使用方法：
```bash
python finetune_plm_optuna.py --task AV_APML --split_type random_split --model_name facebook/esm2_t30_150M_UR50D --fold_seed 0 --n_trials 10 --output_dir ./checkpoints_tune
```

#### 主要参数：
- `--task`: 任务名称（如 AV_APML）
- `--model_name`: 预训练模型名称，支持任意 HuggingFace 模型路径
- `--split_type`: 数据分割类型（random_split, mmseqs2_split），默认 random_split
- `--fold_seed`: 交叉验证种子（0-4），默认 0
- `--n_trials`: Optuna 调优试验次数，默认 20
- `--num_train_epochs`: 训练轮数，默认 30
- `--early_stopping_patience`: 早停耐心值，默认 5
- `--output_dir`: 输出目录，默认 ./checkpoints

#### 工作流程：
1. 使用 Optuna 优化超参数（学习率、权重衰减、批次大小等）
2. 使用最佳超参数训练最终模型
3. 在训练集、验证集、测试集上评估性能
4. 保存调优结果和最终模型

#### 输出：
结果保存到 `{output_dir}/{task}/{split_type}/{model_name}/{fold_seed}/`，包含：
- `best_params_{model_name}.json`: 最佳超参数
- `tuning_trials_{model_name}.csv`: 所有调优试验结果
- `model/`: 使用最佳参数训练的最终模型
- `tokenizer/`: 分词器文件
- `metrics.csv`: 最终模型的评估指标结果（使用 pepbenchmark 评估器）
- `training_config.json`: 包含最佳超参数的训练配置

### 3. run_one_dataset.sh - 批量运行脚本
一次性跑一个数据集的两种划分方式的5个种子：
```bash
bash run_one_dataset.sh AV_APML
```

### 4. run_all.sh 
调用run_one_dataset.sh ,跑所有数据

## 可用模型

支持的预训练模型包括：
- `facebook/esm2_t30_150M_UR50D` (ESM2-150M, 推荐)
- `facebook/esm2_t33_650M_UR50D` (ESM2-650M)
- `facebook/esm2_t36_3B_UR50D` (ESM2-3B)
- `Rostlab/prot_bert_bfd` (ProtBERT)
- 或任意其他 HuggingFace 蛋白质语言模型

**注意**: 两个脚本现在都直接使用完整的 HuggingFace 模型名称，不再支持简化名称映射。

## 建议工作流程

### 方案一：直接训练（推荐）
```bash
python finetune_plm.py --task AV_APML --split_type random_split --model_name facebook/esm2_t30_150M_UR50D
```

### 方案二：先调优再训练（可选）
```bash
# 第一步：超参数调优并训练最终模型
python finetune_plm_optuna.py --task AV_APML --split_type random_split --model_name facebook/esm2_t30_150M_UR50D --n_trials 10

# 注意：第一步已经包含了最终模型训练，通常无需再次运行 finetune_plm.py
```

## 评估系统

两个脚本都使用 `pepbenchmark.evaluator` 进行统一的性能评估：

- **分类任务**: 使用 `evaluate_classification`，支持二分类和多分类
- **回归任务**: 使用 `evaluate_regression`
- **评估指标**: 包括准确率、精确率、召回率、F1分数、AUC、MCC等多种指标

这确保了评估结果的一致性和可比性。
