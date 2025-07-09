# 数据集预处理模块使用指南

## 概述

`pepbenchmark.preprocess` 模块提供了完整的肽数据集预处理功能。它能够：

1. 从 `DATASET_MAP` 中指定数据集
2. 读取数据集路径下的 `combine.csv` 文件（包含 `sequence` 和 `label` 两列）
3. 根据 `OFFICIAL_FEATURE_TYPES` 生成各种特征类型
4. 保存处理后的特征文件
5. 生成训练/验证/测试数据划分

## 支持的特征类型

- **fasta**: 原始氨基酸序列
- **smiles**: SMILES分子表示
- **helm**: HELM分子表示
- **biln**: BiLN分子表示
- **ecfp**: Extended Connectivity Fingerprints (ECFP)
- **fasta_esm2_150**: ESM2模型生成的蛋白质嵌入
- **graph**: 分子图表示
- **label**: 标签数据
- **random_split**: 随机数据划分
- **mmseqs2_split**: 基于同源性的数据划分

## 基本使用方法

### 方法1: 使用便捷函数

```python
from pepbenchmark.preprocess import preprocess_dataset

# 预处理整个数据集（生成所有特征）
preprocessor = preprocess_dataset(
    dataset_name="BBP_APML",
    save_results=True
)

# 只生成特定特征
preprocessor = preprocess_dataset(
    dataset_name="BBP_APML",
    feature_types=["fasta", "smiles", "ecfp"],
    save_results=True
)
```

### 方法2: 手动控制预处理流程

```python
from pepbenchmark.preprocess import DatasetPreprocessor

# 初始化预处理器
preprocessor = DatasetPreprocessor(dataset_name="BBP_APML")

# 加载原始数据
raw_data = preprocessor.load_raw_data()
print(f"加载了 {len(raw_data)} 个样本")

# 生成特定特征
features = preprocessor.generate_features(["fasta", "smiles", "ecfp"])

# 保存特征
preprocessor.save_features()

# 查看特征信息
info = preprocessor.get_feature_info()
print(info)
```

## 文件结构

预处理后，数据集目录将包含以下文件：

```
数据集目录/
├── combine.csv           # 原始数据文件（sequence, label）
├── fasta.csv            # FASTA序列特征
├── smiles.csv           # SMILES分子表示
├── helm.csv             # HELM分子表示
├── biln.csv             # BiLN分子表示
├── ecfp.npz             # ECFP分子指纹（NumPy压缩格式）
├── fasta_esm2_150.npz   # ESM2蛋白质嵌入
├── graph.pt             # 分子图（PyTorch格式）
├── label.csv            # 标签数据
├── random_split.json    # 随机数据划分
└── mmseqs2_split.json   # 基于同源性的数据划分
```

## 数据划分格式

```json
{
  "seed_0": {
    "train": [0, 1, 5, 8, ...],
    "valid": [2, 6, 9, ...],
    "test": [3, 4, 7, ...]
  },
  "seed_1": {
    "train": [...],
    "valid": [...],
    "test": [...]
  },
  ...
}
```

## 高级用法

### 自定义特征生成参数

```python
# 自定义ECFP参数
preprocessor = DatasetPreprocessor("BBP_APML")
preprocessor.load_raw_data()

# 手动配置ECFP生成器
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP

fasta2smiles = Fasta2Smiles()
smiles2fp = Smiles2FP(fp_type="Morgan", radius=4, nBits=4096)

# 生成自定义ECFP
sequences = preprocessor._raw_data["sequence"].tolist()
smiles_list = [fasta2smiles(seq) for seq in sequences]
ecfp_features = [smiles2fp(smiles) for smiles in smiles_list]

# 设置为用户特征
preprocessor.set_user_feature("custom_ecfp", ecfp_features)
```

### 批量处理多个数据集

```python
from pepbenchmark.metadata import DATASET_MAP

# 处理所有ADME相关数据集
adme_datasets = [name for name, info in DATASET_MAP.items()
                 if info.get("group") == "ADME"]

for dataset_name in adme_datasets:
    try:
        print(f"Processing {dataset_name}...")
        preprocess_dataset(
            dataset_name=dataset_name,
            feature_types=["fasta", "smiles", "ecfp", "random_split"],
            save_results=True
        )
        print(f"✓ {dataset_name} processed successfully")
    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {e}")
```

## 命令行使用

```bash
# 处理单个数据集（所有特征）
python -m pepbenchmark.preprocess BBP_APML

# 处理特定特征
python -m pepbenchmark.preprocess BBP_APML --features fasta smiles ecfp

# 强制重新生成
python -m pepbenchmark.preprocess BBP_APML --force
```

## 注意事项

1. **数据文件格式**: `combine.csv` 必须包含 `sequence` 和 `label` 两列
2. **计算资源**: ESM2嵌入和ECFP生成可能需要较长时间和较多内存
3. **依赖项**: 确保安装了所需的依赖包（RDKit, transformers等）
4. **缓存**: 生成的特征会保存到磁盘，避免重复计算

## 错误处理

常见错误及解决方法：

- **FileNotFoundError**: 检查数据集路径和 `combine.csv` 文件是否存在
- **ValueError**: 检查特征类型名称是否正确
- **ImportError**: 安装缺失的依赖包
- **MemoryError**: 减少批处理大小或使用更小的特征维度

## 扩展功能

要添加新的特征类型：

1. 在 `OFFICIAL_FEATURE_TYPES` 中添加新类型
2. 在 `FEATURE_FILE_EXTENSIONS` 中定义文件扩展名
3. 在 `_get_converter` 方法中添加对应的转换器
4. 在 `generate_feature` 方法中添加处理逻辑
