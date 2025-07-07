# 新基类使用示例

这个文档展示了如何使用重写后的BaseSplitter基类。

## 基类功能概览

新的BaseSplitter基类提供了以下功能：

### 1. 抽象方法
- `get_split_indices()`: 必须由子类实现的核心分割方法

### 2. 通用方法
- `get_split_indices_n()`: 生成多个随机分割
- `get_split_kfold_indices()`: 生成k-fold交叉验证分割
- `validate_split_results()`: 验证分割结果
- `get_split_statistics()`: 获取分割统计信息
- `save_split_results()` / `load_split_results()`: 保存/加载分割结果

### 3. 验证方法
- `_validate_fractions()`: 验证分割比例
- `_validate_data()`: 验证输入数据

## 使用示例

### 基本使用
```python
from pepbenchmark.splitter.random_spliter import RandomSplitter
from pepbenchmark.splitter.homo_spliter import MMseqs2Spliter

# 随机分割
random_splitter = RandomSplitter()
splits = random_splitter.get_split_indices(
    data=sequences,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)

# 同源性分割
homo_splitter = MMseqs2Spliter()
splits = homo_splitter.get_split_indices(
    data=sequences,
    identity=0.25,
    coverage=0.8,
    seed=42
)
```

### 多重分割
```python
# 生成5个不同种子的分割
multiple_splits = splitter.get_split_indices_n(
    data=sequences,
    n_splits=5,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)

# 结果格式: {'seed_0': {...}, 'seed_1': {...}, ...}
```

### K-fold交叉验证
```python
# 生成5-fold交叉验证分割
kfold_splits = splitter.get_split_kfold_indices(
    data=sequences,
    k_folds=5,
    seed=42
)

# 结果格式: {'fold_0': {...}, 'fold_1': {...}, ...}
```

### 结果验证
```python
# 验证分割结果
is_valid = splitter.validate_split_results(
    split_results=splits,
    data_size=len(sequences),
    check_completeness=True,
    check_overlaps=True
)

if is_valid:
    print("Split validation passed!")
else:
    print("Split validation failed!")
```

### 获取统计信息
```python
# 获取分割统计
stats = splitter.get_split_statistics(splits)
print(f"Train size: {stats['train_size']} ({stats['train_fraction']:.2%})")
print(f"Valid size: {stats['valid_size']} ({stats['valid_fraction']:.2%})")
print(f"Test size: {stats['test_size']} ({stats['test_fraction']:.2%})")
```

### 保存和加载结果
```python
# 保存分割结果
splitter.save_split_results(
    split_results=splits,
    filepath="data/splits/my_split.json",
    format="json"
)

# 加载分割结果
loaded_splits = splitter.load_split_results(
    filepath="data/splits/my_split.json",
    format="json"
)
```

## 创建自定义分割器

要创建自定义分割器，需要继承BaseSplitter并实现get_split_indices方法：

```python
from pepbenchmark.splitter.base_spliter import BaseSplitter
import numpy as np

class CustomSplitter(BaseSplitter):
    def get_split_indices(self, data, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42, **kwargs):
        self.logger.info(f"Running custom split with seed {seed}")
        self._validate_fractions(frac_train, frac_valid, frac_test)
        self._validate_data(data)

        # 自定义分割逻辑
        n = len(data)
        if seed is not None:
            np.random.seed(seed)

        indices = np.random.permutation(n)

        train_size = int(n * frac_train)
        valid_size = int(n * frac_valid)

        splits = {
            'train': indices[:train_size],
            'valid': indices[train_size:train_size + valid_size],
            'test': indices[train_size + valid_size:]
        }

        # 验证结果
        self.validate_split_results(splits, n)

        return splits
```

## 错误处理

基类提供了完善的错误处理：

```python
try:
    splits = splitter.get_split_indices(
        data=sequences,
        frac_train=0.9,
        frac_valid=0.2,  # 错误：总和超过1.0
        frac_test=0.1
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## 类型提示

新基类支持完整的类型提示：

```python
from typing import List, Dict, Union
import numpy as np

def process_splits(
    splitter: BaseSplitter,
    data: List[str],
    n_splits: int = 5
) -> Dict[str, Dict[str, np.ndarray]]:
    return splitter.get_split_indices_n(data, n_splits=n_splits)
```

## 扩展的SPLIT枚举

新的SPLIT枚举包含更多分割类型：

```python
from pepbenchmark.splitter.base_spliter import SPLIT

print(SPLIT.RANDOM.value)      # "random"
print(SPLIT.STRATIFIED.value)  # "stratified"
print(SPLIT.HOMOLOGY.value)    # "homology"
print(SPLIT.TEMPORAL.value)    # "temporal"
print(SPLIT.CLUSTER.value)     # "cluster"
```

这个重写的基类提供了更强大、更灵活且更易于使用的分割器基础架构。
