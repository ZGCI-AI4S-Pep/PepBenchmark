
# 数据集构建指南

PepBenchmark 提供了灵活的数据集构建和管理系统，支持从完全官方配置到完全自定义的多种使用场景。

数据集构建涉及四个核心要素：

1. **正样本数据** - 具有目标属性的肽序列
2. **负样本数据** - 不具有目标属性的肽序列
3. **特征表示** - 序列的数值化表示（如指纹、嵌入等）
4. **数据划分** - 训练集、验证集、测试集的划分策略

## 使用方式概览

我们提供三种不同灵活程度的数据集构建方式：

### 方式一：完全使用官方数据集
- **适用场景**：快速开始，使用标准基准测试
- **特点**：官方正负样本 + 官方特征 + 官方划分

### 方式二：使用官方样本，自定义特征和划分
- **适用场景**：使用标准数据，但需要自定义特征工程或划分策略
- **特点**：官方正负样本 + 自定义特征 + 自定义划分

### 方式三：完全自定义数据集
- **适用场景**：使用自己的数据或需要特殊的负样本采样策略
- **特点**：自定义正样本 + 自定义负样本 + 自定义特征 + 自定义划分

---

## 方式一：完全使用官方数据集

这是最简单的使用方式，适合快速开始和标准基准测试。
用户可以使用带有`official`的接口：
 — get_official_feature()
 - set_official_feature()
 - get_official_split()
 - set_official_split()
### 基本用法

```python
from pepbenchmark.single_pred import SingleTaskDatasetManager

# 创建数据集管理器并指定要使用的特征
dataset = SingleTaskDatasetManager(
    dataset_name="BBP_APML",  # 血脑屏障穿透肽数据集
    official_feature_names=["fasta", "label"]  # 指定需要的官方特征
)

# 设置官方数据划分（随机划分，种子为0）
dataset.set_official_split_indices(split_type="random_split", fold_seed=0)

# 获取训练、验证、测试数据
train_data, valid_data, test_data = dataset.get_train_val_test_features(format="dict")
```

支持的数据集参考xxx
支持的官方特征参考xxx
支持的划分方式参考xxx
---

## 方式二：使用官方样本，自定义特征和划分

当你想使用官方的正负样本，但需要自定义特征工程或划分策略时使用此方式。

### 自定义特征转换

```python
from pepbenchmark.single_pred import SingleTaskDatasetManager
from pepbenchmark.pep_utils.convert import Fasta2ECFP, Fasta2ESM2
from pepbenchmark.splitter.random_splitter import RandomSplitter

# 加载官方数据集的基础特征
dataset = SingleTaskDatasetManager(
    dataset_name="BBP_APML",
    official_feature_names=["fasta", "label"]
)

# 使用转换器生成自定义特征
fasta_sequences = dataset.get_official_feature("fasta")

# 生成ESM2嵌入
esm2_converter = Fasta2ESM2()
esm2_features = esm2_converter(fasta_sequences)
dataset.set_user_feature("esm2_custom", esm2_features)

# 数据访问


```

### 自定义数据划分

```python
# 使用自定义分割器
splitter = RandomSplitter()
split_indices = splitter.get_split_indices(
    data=fasta_sequences,
    frac_train=0.7,
    frac_valid=0.15,
    frac_test=0.15,
    random_state=42
)

# 设置自定义划分
dataset.set_user_split_indices(split_indices)

# 获取最终数据
train_data, valid_data, test_data = dataset.get_train_val_test_features()
```

### 同源性划分

```python
from pepbenchmark.splitter.homo_splitter import HomologySplitter

# 基于序列同源性的划分，避免数据泄漏
homo_splitter = HomologySplitter(similarity_threshold=0.4)
split_indices = homo_splitter.get_split_indices(
    data=fasta_sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1
)

dataset.set_user_split_indices(split_indices)
```

---

## 方式三：完全自定义数据集

当你有自己的数据或需要特殊的负样本采样策略时使用此方式。

### 使用自定义正样本

```python
from pepbenchmark.single_pred import SingleTaskDatasetManager
from pepbenchmark.pep_utils.redundancy import remove_redundancy
from pepbenchmark.pep_utils.neg_sample import NegSampler

# 准备自定义正样本
positive_sequences = [
    "ALAGGGPCR",
    "KLAGGGPCR",
    "ALAGGGPCK",
    # ... 更多序列
]

# 去除冗余序列
unique_indices = remove_redundancy(
    sequences=positive_sequences,
    similarity_threshold=0.8,
    method="mmseqs2"  # 或 "cd-hit"
)
filtered_sequences = [positive_sequences[i] for i in unique_indices]
```

### 负样本采样

```python
# 创建数据集管理器（用于访问负样本库）
dataset = SingleTaskDatasetManager(dataset_name="BBP_APML")

# 使用负样本采样器
neg_sampler = NegSampler(dataset)
negative_sequences = neg_sampler.sample(
    pos_sequences=filtered_sequences,
    ratio=1.0,  # 负样本与正样本的比例
    method="random",  # 采样方法
    seed=42
)

# 组合正负样本
all_sequences = filtered_sequences + negative_sequences
labels = [1] * len(filtered_sequences) + [0] * len(negative_sequences)
```

### 构建完整数据集

```python
# 创建新的数据集管理器（不加载官方数据）
custom_dataset = SingleTaskDatasetManager(
    dataset_name="custom_dataset",
    official_feature_names=[]  # 不加载官方特征
)

# 设置自定义特征
custom_dataset.set_user_feature("fasta", all_sequences)
custom_dataset.set_user_feature("label", labels)

# 生成其他特征
from pepbenchmark.pep_utils.convert import Fasta2OneHot, Fasta2SMILES

onehot_converter = Fasta2OneHot()
onehot_features = onehot_converter(all_sequences)
custom_dataset.set_user_feature("onehot", onehot_features)

smiles_converter = Fasta2SMILES()
smiles_features = smiles_converter(all_sequences)
custom_dataset.set_user_feature("smiles", smiles_features)

# 设置数据划分
from pepbenchmark.splitter.random_splitter import RandomSplitter

splitter = RandomSplitter()
split_indices = splitter.get_split_indices(
    data=all_sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1
)
custom_dataset.set_user_split_indices(split_indices)
```

---

## 多任务数据集

对于多标签或多任务学习场景：

```python
from pepbenchmark.single_pred import MultiTaskDatasetManager

# 创建多任务数据集
multitask_dataset = MultiTaskDatasetManager(
    dataset_name="multitask_peptidepedia",
    labels=["Antibacterial", "Antiviral"]  # 多个任务标签
)

# 进行负样本采样
multitask_dataset.negative_sampling(ratio=1.0, seed=42)

# 设置自定义特征和划分
# ... (类似单任务流程)
```

---
