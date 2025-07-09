# 安装指南

本章介绍如何安装和配置 PepBenchmark 环境。

## 安装方式

### 方式一：从 PyPI 安装 (推荐)

```bash
pip install pepbenchmark
```

### 方式二：从源码安装

```bash
git clone https://github.com/ZGCI-AI4S-Pep/PepBenchmark.git
cd PepBenchmark
conda env create -f environment.yaml
conda activate pepbenchmark
pip install -e .
```



## 可选组件安装




## 验证安装

### 基本功能测试

```python
import pepbenchmark as pb

# 检查版本
print(f"PepBenchmark version: {pb.__version__}")

# 测试基本转换功能
from pepbenchmark.pep_utils.convert import Fasta2Smiles
converter = Fasta2Smiles()
smiles = converter("ALAGGGPCR")
print(f"SMILES: {smiles}")
```

### 数据集加载测试

```python
from pepbenchmark.metadata import get_all_datasets
# 查看可用数据集
datasets = get_all_datasets()
print(f"Available datasets: {len(datasets)}")
```


## 常见问题




## 下一步

安装完成后，您可以：

1. 阅读[快速开始](quickstart.md)教程
2. 查看[示例代码](examples/)
3. 浏览[API文档](api/modules.rst)
