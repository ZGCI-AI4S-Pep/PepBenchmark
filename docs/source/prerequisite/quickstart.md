# 快速开始

本教程将通过几个简单的例子快速介绍 PepBenchmark 的主要功能。

## 基本使用流程

PepBenchmark 的典型使用流程包括：
1. 数据加载和预处理
2. 分子表示生成
3. 模型训练和评估
4. 结果分析和可视化

## 示例 1: 肽序列格式转换

```python
import pepbenchmark as pb
from pepbenchmark.pep_utils.convert import (
    Fasta2Smiles, Fasta2Helm, Smiles2FP, Fasta2Embedding
)

# 肽序列
peptide = "ALAGGGPCR"
peptide_list = ["ALAGGGPCR", "KLLLLLKLLLKLLLKLLK", "GHRP"]

# 1. FASTA 转 SMILES
fasta2smiles = Fasta2Smiles()
smiles = fasta2smiles(peptide)
print(f"SMILES: {smiles}")

# 批量转换
smiles_list = fasta2smiles(peptide_list)
print(f"Batch SMILES: {smiles_list}")

# 2. FASTA 转 HELM 格式
fasta2helm = Fasta2Helm()
helm = fasta2helm(peptide)
print(f"HELM: {helm}")

# 3. 生成分子指纹
fp_generator = Smiles2FP(fp_type='Morgan', radius=2, nBits=2048)
fingerprint = fp_generator(smiles)
print(f"Fingerprint shape: {fingerprint.shape}")
print(f"Non-zero bits: {fingerprint.sum()}")

# 4. 生成神经网络嵌入
embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
embedding = embedder(peptide)
print(f"Embedding shape: {embedding.shape}")
```

## 示例 2: 数据集加载和基本分析

```python
from pepbenchmark.metadata import natural_binary_keys, get_dataset_info
from pepbenchmark.single_pred.dataset import PeptideDataset
import pandas as pd

# 查看可用的数据集
print("Available natural binary datasets:")
for key in natural_binary_keys[:5]:  # 显示前5个
    info = get_dataset_info(key)
    print(f"  {key}: {info.get('description', 'No description')}")

# 加载特定数据集
dataset_key = "BBP_APML"  # 血脑屏障穿透肽
dataset = PeptideDataset(dataset_key)

print(f"\nDataset: {dataset_key}")
print(f"Size: {len(dataset)}")
print(f"Features: {dataset.get_feature_names()}")

# 查看数据样本
df = dataset.to_dataframe()
print(f"\nData preview:")
print(df.head())

print(f"\nLabel distribution:")
print(df['label'].value_counts())
```

## 示例 3: 模型训练和评估

```python
from pepbenchmark.evaluator import BinaryClassificationEvaluator
from pepbenchmark.splitter import RandomSplitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 准备数据
dataset = PeptideDataset("BBP_APML")
data = dataset.get_features("Morgan")  # 使用Morgan指纹
labels = dataset.get_labels()

# 2. 数据分割
splitter = RandomSplitter(test_size=0.2, random_state=42)
train_idx, test_idx = splitter.split(data, labels)

X_train, X_test = data[train_idx], data[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

# 3. 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 模型训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. 预测
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# 6. 评估
evaluator = BinaryClassificationEvaluator()
metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba)

print("Evaluation Results:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## 示例 4: 数据预处理和冗余去除

```python
from pepbenchmark.pep_utils.redundancy import remove_redundancy
from pepbenchmark.pep_utils.negative_sampling import NegativeSampler

# 1. 冗余去除
sequences = ["ALAGGGPCR", "ALAGGGPCR", "KLLLLLKLLL", "KLLLLKLLL"]  # 包含相似序列
labels = [1, 1, 0, 0]

# 使用MMseqs2去除冗余(需要安装MMseqs2)
try:
    unique_seqs, unique_labels = remove_redundancy(
        sequences, labels,
        similarity_threshold=0.8,
        method="mmseqs2"
    )
    print(f"Original: {len(sequences)}, After deduplication: {len(unique_seqs)}")
except Exception as e:
    print(f"Redundancy removal failed: {e}")
    unique_seqs, unique_labels = sequences, labels

# 2. 负样本采样
positive_seqs = [seq for seq, label in zip(unique_seqs, unique_labels) if label == 1]

sampler = NegativeSampler(method="random_shuffle")
negative_seqs = sampler.generate(positive_seqs, n_samples=len(positive_seqs))

print(f"Generated {len(negative_seqs)} negative samples")
print(f"Example negative: {negative_seqs[0]}")
```

## 示例 5: 可视化分析

```python
from pepbenchmark.visualization import plot_confusion_matrix, plot_roc_curve
from pepbenchmark.visualization import plot_sequence_length_distribution
import matplotlib.pyplot as plt

# 使用前面训练的模型结果
dataset = PeptideDataset("BBP_APML")
sequences = dataset.get_sequences()

# 1. 绘制混淆矩阵
plot_confusion_matrix(y_test, y_pred,
                     title=f"Confusion Matrix - {dataset_key}")
plt.show()

# 2. 绘制ROC曲线
plot_roc_curve(y_test, y_pred_proba,
               title=f"ROC Curve - {dataset_key}")
plt.show()

# 3. 序列长度分布
plot_sequence_length_distribution(sequences,
                                 title=f"Sequence Length Distribution - {dataset_key}")
plt.show()

# 4. 特征重要性
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-20:]  # 前20个重要特征

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_indices)), feature_importance[top_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title('Top 20 Feature Importances')
plt.show()
```

## 示例 6: 批量数据集评估

```python
from pepbenchmark.metadata import natural_binary_keys
from pepbenchmark.utils.benchmark import BenchmarkRunner

# 选择几个数据集进行基准测试
test_datasets = natural_binary_keys[:3]  # 前3个数据集

# 定义要测试的模型
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': None  # 可以添加其他模型
}

# 运行基准测试
runner = BenchmarkRunner(
    datasets=test_datasets,
    models=models,
    feature_types=['Morgan', 'MACCS'],  # 测试不同特征
    cv_folds=5
)

results = runner.run()
print("Benchmark Results:")
print(results.summary())
```

## 主要概念说明

### 1. 数据集类型
- **Natural Binary**: 天然肽的二分类任务
- **Synthetic Binary**: 合成肽的二分类任务
- **Natural Regression**: 天然肽的回归任务
- **Synthetic Regression**: 合成肽的回归任务

### 2. 分子表示方法
- **FASTA**: 标准氨基酸序列格式
- **SMILES**: 分子结构的线性表示
- **HELM**: 大分子的层次化编辑语言
- **Fingerprints**: 分子指纹(Morgan, MACCS等)
- **Embeddings**: 神经网络生成的密集向量

### 3. 评估指标
- **分类**: Accuracy, Precision, Recall, F1, AUC, MCC等
- **回归**: MAE, MSE, R², Pearson/Spearman相关系数等

## 下一步

现在您已经了解了 PepBenchmark 的基本使用方法，可以：

1. 查看[用户指南](user_guide/index.md)了解更多高级功能
2. 浏览[数据集文档](datasets/overview.md)了解所有可用数据集
3. 参考[API文档](api/modules.rst)获取详细的函数说明
4. 查看[示例项目](examples/)获取完整的应用案例

## 常见问题

**Q: 如何添加自定义数据集？**
A: 参考[数据集构建指南](construct_dataset.md)

**Q: 如何使用GPU加速？**
A: 在创建嵌入生成器时指定device参数：
```python
embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D", device="cuda")
```

**Q: 如何处理内存不足问题？**
A: 使用批处理和适当的batch_size：
```python
# 分批处理大型数据集
for batch in dataset.get_batches(batch_size=32):
    # 处理每个批次
    pass
```
