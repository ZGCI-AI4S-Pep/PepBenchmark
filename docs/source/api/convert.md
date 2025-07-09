 O# convert 模块 API 参考

`pepbenchmark.pep_utils.convert` 模块提供了肽序列和分子表示之间的格式转换功能。

## 概览

该模块实现了多种分子表示格式之间的相互转换，包括：

- **FASTA**: 标准氨基酸序列格式
- **SMILES**: 简化分子线性输入规范
- **HELM**: 大分子层次化编辑语言
- **BiLN**: 生物线性表示法
- **Fingerprints**: 分子指纹
- **Embeddings**: 神经网络嵌入

## 基类
c
### FormatTransform

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.FormatTransform
   :members:
   :undoc-members:
   :show-inheritance:

   所有格式转换类的基类，定义了统一的接口和批处理功能。

   **主要方法:**

   - ``__call__(inputs, **kwargs)``: 执行格式转换
   - ``_handle_batch(inputs, **kwargs)``: 处理批量转换
   - ``_process_single(input_item, **kwargs)``: 处理单个输入项
```

## FASTA 转换类

### Fasta2Smiles

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Fasta2Smiles
   :members:
   :undoc-members:
   :show-inheritance:

   将FASTA格式的肽序列转换为SMILES分子表示。

   **示例:**

   >>> converter = Fasta2Smiles()
   >>> smiles = converter("ALAGGGPCR")
   >>> print(smiles)

   >>> # 批量转换
   >>> smiles_list = converter(["ALAGGGPCR", "PEPTIDE"])
```

### Fasta2Helm

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Fasta2Helm
   :members:
   :undoc-members:
   :show-inheritance:

   将FASTA格式转换为HELM(Hierarchical Editing Language for Macromolecules)表示。

   **示例:**

   >>> converter = Fasta2Helm()
   >>> helm = converter("ALAGGGPCR")
```

### Fasta2Biln

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Fasta2Biln
   :members:
   :undoc-members:
   :show-inheritance:

   将FASTA格式转换为BiLN(Biological Linear Notation)表示。
```

### Fasta2Embedding

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Fasta2Embedding
   :members:
   :undoc-members:
   :show-inheritance:

   使用预训练的蛋白质语言模型生成序列嵌入。

   **参数:**

   - ``model (str or torch.nn.Module)``: HuggingFace模型标识符或PyTorch模型实例
   - ``device (str, optional)``: 计算设备，默认为GPU(如果可用)，否则为CPU
   - ``pooling (str, optional)``: 池化策略，支持'mean'、'max'或'cls'，默认为'mean'

   **示例:**

   >>> # 使用ESM-2模型
   >>> embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
   >>> embedding = embedder("ALAGGGPCR")
   >>> print(embedding.shape)  # (640,)

   >>> # 指定设备和池化策略
   >>> embedder = Fasta2Embedding(
   ...     "facebook/esm2_t30_150M_UR50D",
   ...     device="cuda",
   ...     pooling="mean"
   ... )
```

## SMILES 转换类

### Smiles2FP

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Smiles2FP
   :members:
   :undoc-members:
   :show-inheritance:

   从SMILES字符串生成分子指纹。

   **支持的指纹类型:**

   - ``Morgan``: Morgan圆形指纹
   - ``RDKit``: RDKit拓扑指纹
   - ``MACCS``: MACCS结构键
   - ``TopologicalTorsion``: 拓扑扭转指纹
   - ``AtomPair``: 原子对指纹

   **参数:**

   - ``fp_type (str)``: 指纹类型，默认为"Morgan"
   - ``**kwargs``: 指纹特异性参数

   **Morgan指纹参数:**

   - ``radius (int)``: 圆形半径，默认为2
   - ``nBits (int)``: 位数，默认为2048

   **示例:**

   >>> # 默认Morgan指纹
   >>> fp_gen = Smiles2FP()
   >>> fingerprint = fp_gen("CCO")

   >>> # 自定义Morgan参数
   >>> fp_gen = Smiles2FP(fp_type='Morgan', radius=3, nBits=4096)
   >>> fingerprint = fp_gen("CCO")

   >>> # MACCS指纹
   >>> maccs_gen = Smiles2FP(fp_type='MACCS')
   >>> maccs_fp = maccs_gen("CCO")
```

### Smiles2Graph

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Smiles2Graph
   :members:
   :undoc-members:
   :show-inheritance:

   将SMILES字符串转换为图神经网络可用的图表示。

   **示例:**

   >>> converter = Smiles2Graph()
   >>> graph = converter("CCO")  # 乙醇
   >>> print(graph.x.shape)  # 节点特征
   >>> print(graph.edge_index.shape)  # 边索引
```

### Smiles2Fasta

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Smiles2Fasta
   :members:
   :undoc-members:
   :show-inheritance:

   将SMILES表示转换回FASTA序列(当前未实现)。

   .. note::
      此功能当前未实现，返回空字符串。SMILES到序列的转换需要复杂的分子分析算法。
```

## HELM 转换类

### Helm2Fasta

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Helm2Fasta
   :members:
   :undoc-members:
   :show-inheritance:

   将HELM表示转换为FASTA序列格式。
```

### Helm2Smiles

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Helm2Smiles
   :members:
   :undoc-members:
   :show-inheritance:

   将HELM表示转换为SMILES分子表示。
```

### Helm2Biln

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Helm2Biln
   :members:
   :undoc-members:
   :show-inheritance:

   将HELM表示转换为BiLN格式。
```

## BiLN 转换类

### Biln2Fasta

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Biln2Fasta
   :members:
   :undoc-members:
   :show-inheritance:

   将BiLN表示转换为FASTA序列格式。
```

### Biln2Smiles

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Biln2Smiles
   :members:
   :undoc-members:
   :show-inheritance:

   将BiLN表示转换为SMILES分子表示。
```

### Biln2Helm

```{eval-rst}
.. autoclass:: pepbenchmark.pep_utils.convert.Biln2Helm
   :members:
   :undoc-members:
   :show-inheritance:

   将BiLN表示转换为HELM格式。
```

## 使用示例

### 基本转换流程

```python
from pepbenchmark.pep_utils.convert import (
    Fasta2Smiles, Smiles2FP, Fasta2Embedding
)

# 肽序列
peptide = "ALAGGGPCR"

# 步骤1: FASTA转SMILES
fasta2smiles = Fasta2Smiles()
smiles = fasta2smiles(peptide)

# 步骤2: SMILES转分子指纹
fp_generator = Smiles2FP(fp_type='Morgan', radius=2, nBits=2048)
fingerprint = fp_generator(smiles)

# 步骤3: 生成神经嵌入
embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D")
embedding = embedder(peptide)

print(f"SMILES: {smiles}")
print(f"指纹形状: {fingerprint.shape}")
print(f"嵌入形状: {embedding.shape}")
```

### 批量处理

```python
# 批量序列
peptides = ["ALAGGGPCR", "KLLLLLKLLLKLLLKLLK", "GHRP"]

# 批量转换
smiles_list = fasta2smiles(peptides)
fingerprints = fp_generator(smiles_list)
embeddings = embedder(peptides)

print(f"处理了 {len(peptides)} 个序列")
print(f"指纹数组形状: {fingerprints.shape}")
print(f"嵌入数组形状: {len(embeddings)}")
```

### 多种指纹类型比较

```python
from pepbenchmark.pep_utils.convert import Smiles2FP

smiles = "CCO"  # 乙醇

# 测试不同指纹类型
fp_types = ['Morgan', 'RDKit', 'MACCS', 'TopologicalTorsion', 'AtomPair']

for fp_type in fp_types:
    try:
        fp_gen = Smiles2FP(fp_type=fp_type)
        fingerprint = fp_gen(smiles)
        print(f"{fp_type:18}: {len(fingerprint)} bits, {fingerprint.sum()} set")
    except Exception as e:
        print(f"{fp_type:18}: Error - {e}")
```

## 性能考虑

### 内存优化

对于大批量数据，建议分批处理：

```python
def process_large_batch(sequences, batch_size=100):
    """分批处理大量序列"""
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_results = converter(batch)
        results.extend(batch_results)

    return results
```

### GPU 加速

嵌入生成可以利用GPU加速：

```python
# 检查GPU可用性
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用GPU生成嵌入
embedder = Fasta2Embedding(
    "facebook/esm2_t30_150M_UR50D",
    device=device
)
```

## 错误处理

转换过程中可能遇到的常见错误：

1. **无效序列**: 包含非标准氨基酸的FASTA序列
2. **无效SMILES**: 不符合SMILES语法的字符串
3. **内存不足**: 处理大批量数据时的内存限制
4. **模型加载失败**: 网络问题或模型文件损坏

建议使用try-except块处理这些异常：

```python
try:
    smiles = fasta2smiles(peptide)
    fingerprint = fp_generator(smiles)
except Exception as e:
    print(f"转换失败: {e}")
    # 提供默认值或跳过该样本
```

## 相关模块

- [evaluator](evaluator.md) - 模型评估指标
- [splitter](splitter.md) - 数据分割策略
- [preprocess](preprocess.md) - 数据预处理工具
