# NegSampler类文档

`NegSampler`类是PepBenchmark中用于负样本采样的核心工具。它为分类任务提供智能的负样本生成策略，确保负样本与正样本在属性分布上保持平衡。

## 概述

`NegSampler`类支持以下功能：

- **智能负样本采样**：基于属性分布平衡的采样策略
- **多种采样池**：支持官方采样池和自定义采样池
- **属性匹配**：确保负样本与正样本在关键属性上匹配
- **可视化分析**：生成采样前后的属性分布对比图
- **批量处理**：支持大规模数据集的负样本采样

## 初始化

```python
from pepbenchmark.pep_utils.neg_sample import NegSampler

# 使用官方采样池
sampler = NegSampler(
    dataset_name="BBP",
    user_sampling_pool_path=None,
    filt_length=50,
    dedup_identity=0.9,
    processes=4
)

# 使用自定义采样池
sampler = NegSampler(
    dataset_name="BBP",
    user_sampling_pool_path="/path/to/custom_neg_pool.csv",
    filt_length=50,
    dedup_identity=0.9,
    processes=4
)
```

## 主要方法

### 1. 负样本采样

```python
def get_sample_result(
    self,
    fasta: List[str],
    ratio: float,
    limit: Optional[str] = None,
    seed: int = 42
) -> List[str]:
    """
    获取负样本采样结果

    Args:
        fasta: 正样本序列列表
        ratio: 负样本与正样本的比例
        limit: 限制条件 ("length", "property")
        seed: 随机种子

    Returns:
        负样本序列列表
    """
```

**示例：**

```python
# 基本采样
pos_seqs = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK"]
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,  # 负样本数量等于正样本数量
    limit="length",  # 基于长度限制
    seed=42
)
print(f"采样了 {len(neg_seqs)} 个负样本")
```

### 2. 采样信息分析

```python
def get_sample_info(self, properties: Optional[List[str]] = None) -> pd.DataFrame:
    """
    获取采样信息统计

    Args:
        properties: 要分析的属性列表

    Returns:
        包含统计信息的DataFrame
    """
```

**示例：**

```python
# 获取采样信息
info_df = sampler.get_sample_info(properties=["length", "charge", "hydrophobicity"])
print(info_df)
```

### 3. 可视化分析

```python
def visualization(
    self,
    properties: Optional[List[str]] = None,
    plot_type: str = "kde",
    bins: int = 20,
):
    """
    生成采样前后的属性分布可视化

    Args:
        properties: 要可视化的属性列表
        plot_type: 图表类型 ("kde", "histogram")
        bins: 直方图分箱数
    """
```

**示例：**

```python
# 生成可视化图表
sampler.visualization(
    properties=["length", "charge", "hydrophobicity"],
    plot_type="kde",
    bins=20
)
```

## 支持的采样策略

### 1. 基于属性的采样

`NegSampler`支持基于多种属性的智能采样：

- **长度匹配**：确保负样本与正样本长度分布相似
- **电荷匹配**：匹配正负样本的电荷分布
- **疏水性匹配**：匹配疏水性分布
- **氨基酸组成匹配**：匹配氨基酸组成分布

### 2. 采样池类型

#### 官方采样池

```python
# 使用官方采样池
sampler = NegSampler(dataset_name="BBP")
neg_seqs = sampler.get_sample_result(pos_seqs, ratio=1.0)
```

支持的官方采样池：
- `bbp`: 血脑屏障穿透肽负样本
- `antibacterial`: 抗菌肽负样本
- `antifungal`: 抗真菌肽负样本
- `antiviral`: 抗病毒肽负样本
- `anticancer`: 抗癌肽负样本
- `hemolytic`: 溶血肽负样本
- `toxicity`: 毒性肽负样本

#### 自定义采样池

```python
# 使用自定义采样池
sampler = NegSampler(
    dataset_name="BBP",
    user_sampling_pool_path="/path/to/custom_neg_pool.csv"
)
neg_seqs = sampler.get_sample_result(pos_seqs, ratio=1.0)
```

自定义采样池文件格式：
```csv
sequence,length,charge,hydrophobicity
KLLLLLKLLL,10,1,0.5
ALAGGGPCR,9,0,0.3
...
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | - | 数据集名称 |
| `user_sampling_pool_path` | str | None | 自定义采样池路径 |
| `filt_length` | int | None | 长度过滤阈值 |
| `dedup_identity` | float | None | 去重相似度阈值 |
| `processes` | int | None | 并行进程数 |
| `ratio` | float | 1.0 | 负样本与正样本的比例 |
| `limit` | str | None | 限制条件 ("length", "property") |
| `seed` | int | 42 | 随机种子 |

## 采样策略详解

### 1. 基于长度的采样

```python
# 基于长度限制的采样
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="length",
    seed=42
)
```

### 2. 基于属性的采样

```python
# 基于属性匹配的采样
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="property",
    seed=42
)
```

### 3. 多属性平衡采样

```python
# 多属性平衡采样
sampler = NegSampler(dataset_name="BBP")
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="property",
    seed=42
)

# 分析采样效果
sampler.get_sample_info(properties=["length", "charge", "hydrophobicity"])
sampler.visualization(properties=["length", "charge", "hydrophobicity"])
```

## 完整示例

### 基本负样本采样

```python
from pepbenchmark.pep_utils.neg_sample import NegSampler

# 准备正样本
pos_seqs = [
    "ALAGGGPCR",
    "KLAGGGPCR",
    "ALAGGGPCK",
    "KLLLLLKLLL"
]

# 创建采样器
sampler = NegSampler(
    dataset_name="BBP",
    filt_length=50,
    dedup_identity=0.9,
    processes=4
)

# 采样负样本
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="length",
    seed=42
)

print(f"正样本数量: {len(pos_seqs)}")
print(f"负样本数量: {len(neg_seqs)}")
print(f"采样比例: {len(neg_seqs)/len(pos_seqs):.2f}")
```

### 属性匹配采样

```python
# 使用属性匹配策略
sampler = NegSampler(dataset_name="BBP")

# 采样负样本
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="property",
    seed=42
)

# 分析采样效果
info_df = sampler.get_sample_info(properties=["length", "charge", "hydrophobicity"])
print("采样信息:")
print(info_df)

# 生成可视化
sampler.visualization(
    properties=["length", "charge", "hydrophobicity"],
    plot_type="kde"
)
```

### 自定义采样池

```python
import pandas as pd

# 创建自定义负样本池
custom_neg_pool = pd.DataFrame({
    "sequence": [
        "KLLLLLKLLL",
        "ALAGGGPCR",
        "KLAGGGPCR",
        "KLLLLKLLL",
        "ALAGGGPCK"
    ],
    "length": [10, 9, 9, 9, 9],
    "charge": [1, 0, 1, 1, 0],
    "hydrophobicity": [0.5, 0.3, 0.4, 0.6, 0.2]
})

# 保存自定义采样池
custom_neg_pool.to_csv("custom_neg_pool.csv", index=False)

# 使用自定义采样池
sampler = NegSampler(
    dataset_name="BBP",
    user_sampling_pool_path="custom_neg_pool.csv"
)

neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="property",
    seed=42
)
```

### 批量处理

```python
import pandas as pd
from pepbenchmark.pep_utils.neg_sample import NegSampler

# 加载正样本数据
pos_df = pd.read_csv("pos_seqs.csv")
pos_seqs = pos_df["sequence"].tolist()

# 创建采样器
sampler = NegSampler(
    dataset_name="BBP",
    filt_length=50,
    dedup_identity=0.9,
    processes=8
)

# 批量采样
neg_seqs = sampler.get_sample_result(
    fasta=pos_seqs,
    ratio=1.0,
    limit="property",
    seed=42
)

# 保存结果
result_df = pd.DataFrame({
    "sequence": neg_seqs,
    "label": [0] * len(neg_seqs)  # 负样本标签为0
})
result_df.to_csv("negative_samples.csv", index=False)
```

## 性能优化建议

### 1. 并行处理

```python
# 使用多进程加速
sampler = NegSampler(
    dataset_name="BBP",
    processes=8  # 根据CPU核心数调整
)
```

### 2. 预过滤

```python
# 预先过滤采样池
sampler = NegSampler(
    dataset_name="BBP",
    filt_length=50,      # 长度过滤
    dedup_identity=0.9   # 去重过滤
)
```

### 3. 缓存机制

`NegSampler`实现了智能缓存机制：
- 采样池缓存：避免重复加载
- 属性计算缓存：避免重复计算
- 采样结果缓存：避免重复采样

## 错误处理

### 常见错误及解决方案

1. **采样池文件不存在**
   ```python
   # 错误信息：File not found
   # 解决方案：检查文件路径是否正确
   sampler = NegSampler(
       dataset_name="BBP",
       user_sampling_pool_path="/correct/path/to/neg_pool.csv"
   )
   ```

2. **内存不足**
   ```python
   # 解决方案：减少并行进程数
   sampler = NegSampler(dataset_name="BBP", processes=2)
   ```

3. **采样比例过高**
   ```python
   # 解决方案：降低采样比例或增加采样池大小
   neg_seqs = sampler.get_sample_result(
       fasta=pos_seqs,
       ratio=0.5,  # 降低比例
       limit="length"
   )
   ```

## 相关文档

- [数据预处理指南](./preprocessing.md) - 完整的数据预处理流程
- [Redundancy类文档](./redundancy.md) - 冗余去除功能
- [数据划分文档](./data_splitting.md) - 数据划分策略
