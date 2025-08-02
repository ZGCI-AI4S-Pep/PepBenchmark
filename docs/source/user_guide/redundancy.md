# Redundancy类文档

`Redundancy`类是PepBenchmark中用于冗余分析和过滤的核心工具。它提供了多种方法来检测和去除肽序列中的冗余，确保数据集的多样性和质量。

## 概述

`Redundancy`类支持以下功能：

- **相似性分析**：计算序列间的相似性分布
- **冗余去除**：使用多种算法去除相似序列
- **可视化**：生成相似性分布图表
- **缓存机制**：避免重复计算相似性

## 初始化

```python
from pepbenchmark.pep_utils.redundancy import Redundancy

# 创建Redundancy实例
redundancy = Redundancy()
```

## 主要方法

### 1. 相似性分析

```python
def analyze(self, sequences: List[str], threshold: float = 0.9, processes: int = 16) -> None:
    """
    分析序列的相似性分布

    Args:
        sequences: 序列列表
        threshold: 相似性阈值
        processes: 并行进程数
    """
```

**示例：**

```python
sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK", "KLLLLLKLLL"]
redundancy.analyze(sequences, threshold=0.8, processes=4)
```

### 2. 冗余去除

```python
def deduplicate(
    self,
    sequences: List[str],
    threshold: float = 0.9,
    dedup_method: str = "aar",
    identity: float = 0.9,
    processes: int = 16,
    visualization: bool = True,
) -> List[str]:
    """
    去除冗余序列

    Args:
        sequences: 输入序列列表
        threshold: 相似性阈值
        dedup_method: 去重方法 ("mmseqs", "cdhit", "aar")
        identity: 序列相似度阈值
        processes: 并行进程数
        visualization: 是否生成可视化图表

    Returns:
        去重后的序列列表
    """
```

**示例：**

```python
# 使用MMseqs2进行去重
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9,
    processes=4,
    visualization=True
)
print(f"原始序列数: {len(sequences)}")
print(f"去重后序列数: {len(remain_seqs)}")
```

## 支持的去重方法

### 1. MMseqs2方法

- **优点**：速度快，内存效率高
- **适用场景**：大规模数据集
- **参数**：`identity` (相似度阈值)

```python
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9
)
```

### 2. CD-HIT方法

- **优点**：精确度高，结果可靠
- **适用场景**：需要高精度去重的场景
- **参数**：`identity` (相似度阈值)

```python
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="cdhit",
    identity=0.9
)
```

### 3. AAR方法 (Amino Acid Residue)

- **优点**：基于氨基酸残差的快速去重
- **适用场景**：快速初步去重
- **参数**：`threshold` (相似性阈值)

```python
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="aar",
    threshold=0.9
)
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sequences` | List[str] | - | 输入序列列表 |
| `threshold` | float | 0.9 | 相似性阈值 |
| `dedup_method` | str | "aar" | 去重方法 ("mmseqs", "cdhit", "aar") |
| `identity` | float | 0.9 | 序列相似度阈值 |
| `processes` | int | 16 | 并行进程数 |
| `visualization` | bool | True | 是否生成可视化图表 |

## 缓存机制

`Redundancy`类实现了智能缓存机制，避免重复计算相似性：

- **数据哈希**：基于序列内容生成哈希值
- **阈值匹配**：检查当前阈值是否与缓存匹配
- **自动更新**：数据或参数变化时自动重新计算

## 可视化功能

### 相似性分布图

```python
# 分析相似性分布
redundancy.analyze(sequences, threshold=0.9)

# 去重并生成可视化
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9,
    visualization=True
)
```

生成的可视化图表包括：
- 去重前后的相似性分布对比
- 相似性分数分布直方图
- 去重效果统计信息

## 完整示例

### 基本使用

```python
from pepbenchmark.pep_utils.redundancy import Redundancy

# 准备数据
sequences = [
    "ALAGGGPCR",
    "KLAGGGPCR",
    "ALAGGGPCK",
    "KLLLLLKLLL",
    "KLLLLKLLL",
    "ALAGGGPCR",  # 重复序列
    "KLAGGGPCR"   # 重复序列
]

# 创建Redundancy实例
redundancy = Redundancy()

# 分析相似性分布
redundancy.analyze(sequences, threshold=0.8, processes=4)

# 使用MMseqs2去重
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9,
    processes=4,
    visualization=True
)

print(f"原始序列数: {len(sequences)}")
print(f"去重后序列数: {len(remain_seqs)}")
print(f"去重率: {(1 - len(remain_seqs)/len(sequences))*100:.2f}%")
```

### 批量处理

```python
import pandas as pd
from pepbenchmark.pep_utils.redundancy import Redundancy

# 加载数据
df = pd.read_csv("pos_seqs.csv")
sequences = df["sequence"].tolist()

# 创建Redundancy实例
redundancy = Redundancy()

# 批量去重
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9,
    processes=8,
    visualization=True
)

# 保存结果
result_df = pd.DataFrame({"sequence": remain_seqs})
result_df.to_csv("deduplicated_seqs.csv", index=False)
```

### 不同方法的比较

```python
from pepbenchmark.pep_utils.redundancy import Redundancy

sequences = ["ALAGGGPCR", "KLAGGGPCR", "ALAGGGPCK", "KLLLLLKLLL"]

redundancy = Redundancy()

# 比较不同去重方法
methods = ["mmseqs", "cdhit", "aar"]
results = {}

for method in methods:
    remain_seqs = redundancy.deduplicate(
        sequences=sequences,
        dedup_method=method,
        identity=0.9 if method in ["mmseqs", "cdhit"] else 0.9,
        threshold=0.9 if method == "aar" else None
    )
    results[method] = {
        "remaining_count": len(remain_seqs),
        "reduction_rate": (1 - len(remain_seqs)/len(sequences))*100
    }

# 打印比较结果
for method, result in results.items():
    print(f"{method}: 剩余 {result['remaining_count']} 个序列, 减少 {result['reduction_rate']:.2f}%")
```

## 性能优化建议

### 1. 并行处理

```python
# 使用多进程加速计算
remain_seqs = redundancy.deduplicate(
    sequences=sequences,
    dedup_method="mmseqs",
    identity=0.9,
    processes=8  # 根据CPU核心数调整
)
```

### 2. 方法选择

- **大规模数据集**：推荐使用 `mmseqs`
- **高精度要求**：推荐使用 `cdhit`
- **快速初步去重**：推荐使用 `aar`

### 3. 阈值设置

- **严格去重**：`identity=0.95` 或更高
- **适中去重**：`identity=0.9`
- **宽松去重**：`identity=0.8` 或更低

## 错误处理

### 常见错误及解决方案

1. **MMseqs2未安装**
   ```python
   # 错误信息：Command 'mmseqs' not found
   # 解决方案：安装MMseqs2
   # Ubuntu/Debian: sudo apt-get install mmseqs2
   # CentOS/RHEL: sudo yum install mmseqs2
   ```

2. **CD-HIT未安装**
   ```python
   # 错误信息：Command 'cd-hit' not found
   # 解决方案：安装CD-HIT
   # pip install cd-hit
   ```

3. **内存不足**
   ```python
   # 解决方案：减少并行进程数
   redundancy.deduplicate(sequences, processes=2)
   ```

## 相关文档

- [数据预处理指南](./preprocessing.md) - 完整的数据预处理流程
- [NegSampler类文档](./neg_sampler.md) - 负样本采样功能
- [数据划分文档](./data_splitting.md) - 数据划分策略
