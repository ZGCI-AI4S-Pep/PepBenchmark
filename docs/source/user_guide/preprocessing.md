# 数据预处理指南

PepBenchmark提供了完整的数据预处理流程，支持分类和回归两种任务类型。本指南将详细介绍如何使用PepBenchmark进行数据预处理。

## 概述

数据预处理是机器学习流程中的关键步骤，PepBenchmark提供了以下核心功能：

- **数据准备**：加载原始数据，过滤长度，处理缺失值
- **冗余去除**：使用多种方法去除相似序列
- **负样本采样**：为分类任务生成负样本
- **异常值检测**：为回归任务检测和移除异常值
- **特征生成**：生成多种类型的特征表示
- **数据划分**：支持多种划分策略

## 支持的数据集类型

PepBenchmark支持两种主要的数据集类型：

### 1. 分类数据集 (Binary Classification)
- 包含正样本序列文件 (`pos_seqs.csv`)
- 需要负样本采样
- 支持冗余去除
- 示例：BBP、Antibacterial、Antifungal等

### 2. 回归数据集 (Regression)
- 包含序列和标签的完整数据 (`origin_data.csv`)
- 支持异常值检测和移除
- 支持数据去重
- 示例：E.coli_mic、P.aeruginosa_mic等

## 分类数据集预处理

### 基本流程

```python
from pepbenchmark.prepare import prepare_processed_class_data

# 预处理分类数据集
prepare_processed_class_data(
    dataset_name="BBP",
    dedup_method="mmseqs",
    dedup_identity=0.9,
    neg_sample_ratio=1.0,
    filter_length=50,
    validate_data=True,
    processes=4,
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./plots",
    random_seed=42
)
```

### 详细步骤

#### 1. 数据加载和过滤

```python
from pepbenchmark.prepare import ClassDataPrepare

# 初始化数据准备器
dataprepare = ClassDataPrepare(
    dataset_name="BBP",
    filter_length=50  # 过滤长度超过50的序列
)

# 加载正样本序列
pos_seqs = dataprepare.load_raw_pos_seqs()
print(f"加载了 {len(pos_seqs)} 个正样本序列")
```

#### 2. 冗余去除

```python
# 去除冗余序列
remain_seqs = dataprepare.filt_redundancy(
    dedup_method="mmseqs",  # 使用MMseqs2进行去重
    identity=0.9,           # 相似度阈值
    processes=4,            # 并行进程数
    visualization=True      # 生成可视化图表
)
print(f"去重后剩余 {len(remain_seqs)} 个序列")
```

#### 3. 负样本采样

```python
# 采样负样本
neg_seqs = dataprepare.sample_neg_seqs(
    user_sampling_pool_path=None,  # 自定义负样本池路径
    filt_length=50,                # 长度过滤
    dedup_identity=0.9,           # 去重相似度阈值
    ratio=1.0,                    # 负样本与正样本的比例
    random_seed=42,               # 随机种子
    processes=4                   # 并行进程数
)
print(f"采样了 {len(neg_seqs)} 个负样本")
```

#### 4. 保存组合数据

```python
# 保存正负样本组合数据
dataprepare.save_combine_csv()
```

#### 5. 数据验证

```python
# 验证数据质量
dataprepare.validate_data(
    processes=4,
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./validation_plots"
)
```

## 回归数据集预处理

### 基本流程

```python
from pepbenchmark.prepare import prepare_processed_regression_data

# 预处理回归数据集
prepare_processed_regression_data(
    dataset_name="E.coli_mic",
    filter_length=50,
    outlier_remove_method="iqr",
    validate_data=True,
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./regression_plots"
)
```

### 详细步骤

#### 1. 数据加载和过滤

```python
from pepbenchmark.prepare import RegressionDataPrepare

# 初始化回归数据准备器
dataprepare = RegressionDataPrepare(
    dataset_name="E.coli_mic",
    filter_length=50
)

# 加载原始数据
seqs, labels = dataprepare.load_raw_data()
print(f"加载了 {len(seqs)} 个序列")
```

#### 2. 异常值检测和移除

```python
# 移除异常值
clean_seqs, clean_labels = dataprepare.remove_outliers(
    method="iqr",        # 使用IQR方法检测异常值
    threshold=1.5,       # IQR倍数阈值
    z_threshold=3.0      # Z-score阈值（当method="zscore"时使用）
)
print(f"移除异常值后剩余 {len(clean_seqs)} 个序列")
```

#### 3. 数据去重

```python
# 去除完全重复的序列
dedup_seqs, dedup_labels = dataprepare.deduplicate()
print(f"去重后剩余 {len(dedup_seqs)} 个序列")
```

#### 4. 保存处理后的数据

```python
# 保存处理后的数据
dataprepare.save_processed_data()
```

#### 5. 数据验证

```python
# 验证数据质量
dataprepare.validate_data(
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./regression_validation_plots"
)
```

## 特征生成

### 支持的特征类型

PepBenchmark支持以下特征类型：

- **fasta**: 原始FASTA序列
- **smiles**: SMILES分子表示
- **helm**: HELM表示
- **biln**: BILN表示
- **ecfp4**: Morgan指纹（半径2）
- **ecfp6**: Morgan指纹（半径3）
- **graph**: 分子图表示
- **esm2_150_embedding**: ESM2-150M嵌入
- **dpml_embedding**: DPML嵌入
- **label**: 标签数据

### 特征生成流程

```python
from pepbenchmark.preprocess import preprocess_dataset

# 生成所有支持的特征
preprocessor = preprocess_dataset(
    dataset_name="BBP",
    feature_types=None,  # None表示生成所有特征
    save_results=True
)

# 生成特定特征
preprocessor = preprocess_dataset(
    dataset_name="BBP",
    feature_types=["fasta", "smiles", "ecfp4", "esm2_150_embedding"],
    save_results=True
)
```

### 手动特征生成

```python
from pepbenchmark.preprocess import DatasetPreprocessor

# 初始化预处理器
preprocessor = DatasetPreprocessor("BBP")

# 加载原始数据
raw_data = preprocessor.load_raw_data()

# 生成特定特征
features = preprocessor.generate_features(["fasta", "smiles", "ecfp4"])

# 保存特征
preprocessor.save_features()
```

## 数据划分

PepBenchmark支持多种数据划分策略：

### 1. 随机划分 (Random Split)

```python
from pepbenchmark.splitter.random_splitter import RandomSplitter

splitter = RandomSplitter()
split_result = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)
```

### 2. MMseqs2划分 (MMseqs2 Split)

```python
from pepbenchmark.splitter.mmseq_splitter import MMseqs2Splitter

splitter = MMseqs2Splitter()
split_result = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    identity=0.4,  # 序列相似度阈值
    seed=42
)
```

### 3. CD-HIT划分 (CD-HIT Split)

```python
from pepbenchmark.splitter.cdhit_splitter import CDHitSplitter

splitter = CDHitSplitter()
split_result = splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    identity=0.4,  # 序列相似度阈值
    seed=42
)
```

## 参数说明

### 分类数据集参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | - | 数据集名称 |
| `dedup_method` | str | "mmseqs" | 去重方法 ("mmseqs", "cdhit", "aar") |
| `dedup_identity` | float | 0.9 | 去重相似度阈值 |
| `neg_sample_ratio` | float | 1.0 | 负样本与正样本的比例 |
| `filter_length` | int | 50 | 序列长度过滤阈值 |
| `validate_data` | bool | False | 是否进行数据验证 |
| `processes` | int | None | 并行进程数 |
| `enable_visualization` | bool | True | 是否生成可视化图表 |
| `save_plots` | bool | True | 是否保存图表 |
| `plot_save_dir` | str | "./plots" | 图表保存目录 |
| `random_seed` | int | 42 | 随机种子 |

### 回归数据集参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | - | 数据集名称 |
| `filter_length` | int | 50 | 序列长度过滤阈值 |
| `outlier_remove_method` | str | "iqr" | 异常值检测方法 ("iqr", "zscore") |
| `validate_data` | bool | True | 是否进行数据验证 |
| `enable_visualization` | bool | True | 是否生成可视化图表 |
| `save_plots` | bool | True | 是否保存图表 |
| `plot_save_dir` | str | "./regression_plots" | 图表保存目录 |

## 输出文件

预处理完成后，会在数据集目录下生成以下文件：

### 分类数据集
- `combine.csv`: 正负样本组合数据
- `{feature_type}.{extension}`: 各种特征文件

### 回归数据集
- `processed_data.csv`: 处理后的数据
- `{feature_type}.{extension}`: 各种特征文件

### 特征文件扩展名
- CSV文件: `.csv`
- NumPy压缩文件: `.npz`
- PyTorch张量文件: `.pt`
- JSON文件: `.json`

## 相关文档

- [Redundancy类文档](./redundancy.md) - 冗余去除功能
- [NegSampler类文档](./neg_sampler.md) - 负样本采样功能
- [数据划分文档](./data_splitting.md) - 数据划分策略
- [特征转换文档](./converter.md) - 特征转换功能

## 示例

### 完整分类数据集预处理示例

```python
from pepbenchmark.prepare import prepare_processed_class_data

# 预处理BBP数据集
prepare_processed_class_data(
    dataset_name="BBP",
    dedup_method="mmseqs",
    dedup_identity=0.9,
    neg_sample_ratio=1.0,
    filter_length=50,
    validate_data=True,
    processes=4,
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./bbp_plots",
    random_seed=42
)
```

### 完整回归数据集预处理示例

```python
from pepbenchmark.prepare import prepare_processed_regression_data

# 预处理E.coli_mic数据集
prepare_processed_regression_data(
    dataset_name="E.coli_mic",
    filter_length=50,
    outlier_remove_method="iqr",
    validate_data=True,
    enable_visualization=True,
    save_plots=True,
    plot_save_dir="./ecoli_plots"
)
```

### 自定义特征生成示例

```python
from pepbenchmark.preprocess import preprocess_dataset

# 生成特定特征
preprocessor = preprocess_dataset(
    dataset_name="BBP",
    feature_types=["fasta", "smiles", "ecfp4", "esm2_150_embedding"],
    save_results=True
)

# 查看生成的特征信息
info = preprocessor.get_feature_info()
print(f"生成的特征: {info['loaded_features']}")
print(f"特征形状: {info['feature_shapes']}")
```
