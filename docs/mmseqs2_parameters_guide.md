# MMseqs2 参数配置指南

这个文档说明了如何在PepBenchmark中使用可配置的MMseqs2参数。

## 默认参数

```python
default_params = {
    'coverage': 0.25,        # 覆盖度阈值
    'sensitivity': 10,       # 灵敏度
    'alignment_mode': 3,     # 对齐模式
    'seq_id_mode': 1,        # 序列身份模式
    'mask': 0,              # 掩码
    'cov_mode': 2,          # 覆盖度模式
}
```

## 使用示例

### 1. 基本使用（使用默认参数）

```python
from pepbenchmark.splitter.homo_spliter import MMseqs2Spliter

splitter = MMseqs2Spliter()
splits = splitter.get_split_indices(
    data=fasta_sequences,
    identity=0.25,  # 序列身份阈值
    seed=42
)
```

### 2. 自定义MMseqs2参数

```python
# 高精度聚类
splits = splitter.get_split_indices(
    data=fasta_sequences,
    identity=0.4,           # 更高的身份阈值
    coverage=0.8,           # 更高的覆盖度要求
    sensitivity=7.5,        # 降低灵敏度以提高精度
    seed=42
)

# 快速聚类（适用于大数据集）
splits = splitter.get_split_indices(
    data=fasta_sequences,
    identity=0.25,
    coverage=0.3,           # 降低覆盖度要求
    sensitivity=4.0,        # 降低灵敏度以提高速度
    threads=8,              # 使用多线程
    seed=42
)

# 严格聚类（最小冗余）
splits = splitter.get_split_kfold_indices(
    data=fasta_sequences,
    k_folds=5,
    identity=0.8,           # 高身份阈值
    coverage=0.9,           # 高覆盖度
    sensitivity=15.0,       # 高灵敏度
    cov_mode=1,            # 严格覆盖度模式
    seed=42
)
```

### 3. 批量处理不同参数

```python
# 测试不同参数组合
parameter_sets = [
    {'identity': 0.25, 'coverage': 0.25, 'sensitivity': 10},
    {'identity': 0.4, 'coverage': 0.5, 'sensitivity': 7.5},
    {'identity': 0.6, 'coverage': 0.8, 'sensitivity': 15},
]

results = {}
for i, params in enumerate(parameter_sets):
    splits = splitter.get_split_indices_n(
        data=fasta_sequences,
        n_splits=5,
        **params,
        seed=42
    )
    results[f'params_set_{i}'] = splits
```

## 参数说明

### 核心参数
- **identity**: 序列身份阈值 (0.0-1.0)
- **coverage**: 覆盖度阈值 (0.0-1.0)
- **sensitivity**: 灵敏度 (1.0-20.0，越高越敏感但越慢)

### 高级参数
- **alignment_mode**: 对齐模式 (0-3)
  - 0: 局部对齐
  - 1: 全局对齐
  - 2: 半全局对齐
  - 3: 自动选择（默认）

- **seq_id_mode**: 序列身份计算模式 (0-2)
  - 0: 基于较短序列
  - 1: 基于较长序列（默认）
  - 2: 基于对齐长度

- **cov_mode**: 覆盖度计算模式 (0-2)
  - 0: 基于目标序列
  - 1: 基于查询序列
  - 2: 基于较短序列（默认）

### 性能参数
- **threads**: 线程数（正整数）
- **mask**: 低复杂度区域掩码 (0或1)
- **max_iterations**: 最大迭代次数

## 推荐设置

### 快速原型开发
```python
# 快速但可能不够精确
params = {
    'identity': 0.25,
    'coverage': 0.25,
    'sensitivity': 4.0,
    'threads': 4
}
```

### 高质量聚类
```python
# 更准确但较慢
params = {
    'identity': 0.4,
    'coverage': 0.6,
    'sensitivity': 10.0,
    'cov_mode': 1
}
```

### 严格去冗余
```python
# 最小冗余，适用于训练集准备
params = {
    'identity': 0.8,
    'coverage': 0.9,
    'sensitivity': 15.0,
    'alignment_mode': 1,
    'seq_id_mode': 1,
    'cov_mode': 1
}
```

## 注意事项

1. **身份阈值**越高，聚类越严格，产生的簇越多
2. **覆盖度**越高，要求序列重叠部分越大
3. **灵敏度**越高，越能检测远程同源性，但计算时间越长
4. 对于短肽序列，建议使用较低的覆盖度阈值
5. 使用多线程可以显著提高大数据集的处理速度

## 性能调优

### 大数据集（>10万序列）
- 使用较低的灵敏度 (4.0-7.0)
- 启用多线程处理
- 考虑预过滤非常短的序列

### 小数据集（<1万序列）
- 可以使用较高的灵敏度 (10.0-15.0)
- 适当提高覆盖度要求以获得更好的聚类质量

### 内存受限环境
- 降低灵敏度
- 减少线程数
- 考虑分批处理
