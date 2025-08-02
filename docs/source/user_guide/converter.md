# 特征转换器文档

PepBenchmark提供了丰富的特征转换功能，支持将肽序列转换为多种不同的表示形式。这些转换器是数据预处理和模型训练的重要组件。

## 概述

PepBenchmark支持以下特征转换器：

- **Fasta2Smiles**: FASTA序列转SMILES分子表示
- **Fasta2Helm**: FASTA序列转HELM表示
- **Fasta2Biln**: FASTA序列转BILN表示
- **Fasta2Embedding**: FASTA序列转嵌入表示
- **Smiles2FP**: SMILES转分子指纹
- **Smiles2Graph**: SMILES转分子图

## 支持的转换类型

### 1. FASTA序列转换器

#### Fasta2Smiles

将FASTA格式的肽序列转换为SMILES分子表示。

```python
from pepbenchmark.pep_utils.convert import Fasta2Smiles

# 创建转换器
converter = Fasta2Smiles()

# 转换序列
sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK"]
smiles_list = converter(sequences)

print("转换结果:")
for seq, smiles in zip(sequences, smiles_list):
    print(f"{seq} -> {smiles}")
```

#### Fasta2Helm

将FASTA序列转换为HELM (Hierarchical Editing Language for Macromolecules) 表示。

```python
from pepbenchmark.pep_utils.convert import Fasta2Helm

# 创建转换器
converter = Fasta2Helm()

# 转换序列
sequences = ["ALAGGGPCR", "KLAGGGPCR"]
helm_list = converter(sequences)

print("HELM表示:")
for seq, helm in zip(sequences, helm_list):
    print(f"{seq} -> {helm}")
```

#### Fasta2Biln

将FASTA序列转换为BILN (Binary Linear Notation) 表示。

```python
from pepbenchmark.pep_utils.convert import Fasta2Biln

# 创建转换器
converter = Fasta2Biln()

# 转换序列
sequences = ["ALAGGGPCR", "KLAGGGPCR"]
biln_list = converter(sequences)

print("BILN表示:")
for seq, biln in zip(sequences, biln_list):
    print(f"{seq} -> {biln}")
```

#### Fasta2Embedding

将FASTA序列转换为嵌入表示，支持多种预训练模型。

```python
from pepbenchmark.pep_utils.convert import Fasta2Embedding

# 创建ESM2嵌入转换器
converter = Fasta2Embedding(model="facebook/esm2_t30_150M_UR50D")

# 转换序列
sequences = ["ALAGGGPCR", "KLAGGGPCR"]
embeddings = converter(sequences)

print(f"嵌入维度: {embeddings[0].shape}")
print(f"嵌入数量: {len(embeddings)}")
```

### 2. SMILES转换器

#### Smiles2FP

将SMILES分子表示转换为分子指纹。

```python
from pepbenchmark.pep_utils.convert import Smiles2FP

# 创建Morgan指纹转换器
converter = Smiles2FP(fp_type="Morgan", radius=2, nBits=2048)

# 转换SMILES
smiles_list = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
fingerprints = converter(smiles_list)

print(f"指纹维度: {fingerprints[0].shape}")
print(f"指纹数量: {len(fingerprints)}")
```

支持的指纹类型：
- **Morgan**: Morgan指纹（ECFP）
- **RDKit**: RDKit指纹
- **MACCS**: MACCS键指纹

#### Smiles2Graph

将SMILES分子表示转换为分子图。

```python
from pepbenchmark.pep_utils.convert import Smiles2Graph

# 创建分子图转换器
converter = Smiles2Graph()

# 转换SMILES
smiles_list = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]
graphs = converter(smiles_list)

print(f"图数量: {len(graphs)}")
print(f"节点特征维度: {graphs[0].x.shape}")
print(f"边特征维度: {graphs[0].edge_index.shape}")
```

## 特征转换流程

### 基本转换流程

```python
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP

# 1. FASTA转SMILES
fasta2smiles = Fasta2Smiles()
sequences = ["ALAGGGPCR", "KLAGGGPCR"]
smiles_list = fasta2smiles(sequences)

# 2. SMILES转分子指纹
smiles2fp = Smiles2FP(fp_type="Morgan", radius=2, nBits=2048)
fingerprints = smiles2fp(smiles_list)

print(f"序列数量: {len(sequences)}")
print(f"SMILES数量: {len(smiles_list)}")
print(f"指纹数量: {len(fingerprints)}")
```

### 嵌入生成流程

```python
from pepbenchmark.pep_utils.convert import Fasta2Embedding

# 创建嵌入转换器
embedding_converter = Fasta2Embedding(
    model="facebook/esm2_t30_150M_UR50D"
)

# 转换序列为嵌入
sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK"]
embeddings = embedding_converter(sequences)

# 处理嵌入结果
for i, (seq, emb) in enumerate(zip(sequences, embeddings)):
    print(f"序列 {i+1}: {seq}")
    print(f"嵌入形状: {emb.shape}")
    print(f"嵌入均值: {emb.mean():.4f}")
    print(f"嵌入标准差: {emb.std():.4f}")
    print()
```

## 支持的嵌入模型

### ESM2模型

```python
# ESM2-150M模型
converter = Fasta2Embedding(model="facebook/esm2_t30_150M_UR50D")

# ESM2-650M模型
converter = Fasta2Embedding(model="facebook/esm2_t33_650M_UR50D")

# ESM2-3B模型
converter = Fasta2Embedding(model="facebook/esm2_t36_3B_UR50D")
```

### 自定义模型

```python
# 使用自定义模型路径
converter = Fasta2Embedding(model="/path/to/custom/model")
```

## 参数说明

### Fasta2Smiles参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sequences` | List[str] | - | 输入序列列表 |

### Fasta2Embedding参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | "facebook/esm2_t30_150M_UR50D" | 预训练模型名称 |
| `device` | str | "cpu" | 计算设备 |
| `batch_size` | int | 32 | 批处理大小 |

### Smiles2FP参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fp_type` | str | "Morgan" | 指纹类型 ("Morgan", "RDKit", "MACCS") |
| `radius` | int | 2 | Morgan指纹半径 |
| `nBits` | int | 2048 | 指纹位数 |

### Smiles2Graph参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `add_hs` | bool | True | 是否添加氢原子 |
| `use_edge_features` | bool | True | 是否使用边特征 |

## 完整示例

### 多特征转换示例

```python
from pepbenchmark.pep_utils.convert import (
    Fasta2Smiles, Fasta2Helm, Fasta2Biln,
    Smiles2FP, Smiles2Graph, Fasta2Embedding
)

# 准备数据
sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK"]

# 1. FASTA转SMILES
fasta2smiles = Fasta2Smiles()
smiles_list = fasta2smiles(sequences)

# 2. FASTA转HELM
fasta2helm = Fasta2Helm()
helm_list = fasta2helm(sequences)

# 3. FASTA转BILN
fasta2biln = Fasta2Biln()
biln_list = fasta2biln(sequences)

# 4. SMILES转Morgan指纹
smiles2fp = Smiles2FP(fp_type="Morgan", radius=2, nBits=2048)
fingerprints = smiles2fp(smiles_list)

# 5. SMILES转分子图
smiles2graph = Smiles2Graph()
graphs = smiles2graph(smiles_list)

# 6. FASTA转嵌入
fasta2embedding = Fasta2Embedding(model="facebook/esm2_t30_150M_UR50D")
embeddings = fasta2embedding(sequences)

# 打印结果
print("转换结果:")
for i, seq in enumerate(sequences):
    print(f"\n序列 {i+1}: {seq}")
    print(f"SMILES: {smiles_list[i]}")
    print(f"HELM: {helm_list[i]}")
    print(f"BILN: {biln_list[i]}")
    print(f"指纹形状: {fingerprints[i].shape}")
    print(f"图节点数: {graphs[i].x.shape[0]}")
    print(f"嵌入形状: {embeddings[i].shape}")
```

### 批量处理示例

```python
import pandas as pd
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP

# 加载数据
df = pd.read_csv("sequences.csv")
sequences = df["sequence"].tolist()

# 创建转换器
fasta2smiles = Fasta2Smiles()
smiles2fp = Smiles2FP(fp_type="Morgan", radius=2, nBits=2048)

# 批量转换
smiles_list = fasta2smiles(sequences)
fingerprints = smiles2fp(smiles_list)

# 保存结果
result_df = pd.DataFrame({
    "sequence": sequences,
    "smiles": smiles_list,
    "fingerprint": [fp.tolist() for fp in fingerprints]
})
result_df.to_csv("converted_features.csv", index=False)
```

### 嵌入生成示例

```python
import torch
from pepbenchmark.pep_utils.convert import Fasta2Embedding

# 创建嵌入转换器
converter = Fasta2Embedding(
    model="facebook/esm2_t30_150M_UR50D",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 转换序列
sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK"]
embeddings = converter(sequences)

# 处理嵌入
embeddings_tensor = torch.stack(embeddings)
print(f"嵌入张量形状: {embeddings_tensor.shape}")

# 计算统计信息
mean_embedding = embeddings_tensor.mean(dim=0)
std_embedding = embeddings_tensor.std(dim=0)
print(f"平均嵌入形状: {mean_embedding.shape}")
print(f"标准差嵌入形状: {std_embedding.shape}")
```

## 性能优化建议

### 1. 批处理

```python
# 使用批处理加速转换
converter = Fasta2Embedding(batch_size=64)
embeddings = converter(sequences)
```

### 2. GPU加速

```python
# 使用GPU加速
converter = Fasta2Embedding(
    model="facebook/esm2_t30_150M_UR50D",
    device="cuda"
)
embeddings = converter(sequences)
```

### 3. 缓存机制

```python
# 转换器会自动缓存结果
converter = Fasta2Smiles()
smiles_list = converter(sequences)  # 第一次转换
smiles_list_cached = converter(sequences)  # 使用缓存
```

## 错误处理

### 常见错误及解决方案

1. **模型下载失败**
   ```python
   # 错误信息：Model not found
   # 解决方案：检查网络连接或使用本地模型
   converter = Fasta2Embedding(model="/local/path/to/model")
   ```

2. **内存不足**
   ```python
   # 解决方案：减少批处理大小
   converter = Fasta2Embedding(batch_size=16)
   ```

3. **序列格式错误**
   ```python
   # 解决方案：检查序列格式
   # 确保序列只包含标准氨基酸字符
   valid_sequences = [seq for seq in sequences if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq)]
   ```

## 相关文档

- [数据预处理指南](./preprocessing.md) - 完整的数据预处理流程
- [Redundancy类文档](./redundancy.md) - 冗余去除功能
- [NegSampler类文档](./neg_sampler.md) - 负样本采样功能
- [数据划分文档](./data_splitting.md) - 数据划分策略
