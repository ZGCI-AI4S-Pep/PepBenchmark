# 🧬 PepBenchmark

> 肽序列分析与基准测试的综合性工具包

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-Enabled-yellow.svg)](.pre-commit-config.yaml)

## 🎯 项目目标

PepBenchmark 致力于为肽研究提供端到端支持，涵盖四个核心任务：

- 🔬 **性质预测 (Property Prediction)** - 预测肽的生物活性、溶解度等性质
- 🏗️ **结构预测 (Structure Prediction)** - 从一级序列推断3D构象
- 🧪 **序列生成 (Sequence Generation)** - 生成具有特定性质的肽序列
- 🎨 **结构生成 (Structure Generation)** - 生成新颖的肽结构

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/your-org/PepBenchmark.git
cd PepBenchmark

# 创建环境
conda create -n pepbenchmark python=3.10
conda activate pepbenchmark

# 安装依赖
pip install -e .
```

### 基础用法

```python
from pepbenchmark.dataset_loader import SingleTaskDataset
from pepbenchmark.pep_utils import PeptideFeaturizer

# 加载数据集
dataset = SingleTaskDataset("BBP_APML")
data = dataset.get_data()

# 特征提取
featurizer = PeptideFeaturizer(
    input_format="fasta",
    feature_type="onehot"
)
features = featurizer.extract_features(data['sequence'])
```

## 📁 项目结构

```
PepBenchmark/
├── 📁 src/pepbenchmark/           # 核心源代码
│   ├── 📁 dataset_loader/         # 数据集加载器
│   ├── 📁 pep_utils/             # 肽序列处理工具
│   ├── 📁 utils/                 # 通用工具
│   ├── 📁 visualization/         # 数据可视化
│   └── 📄 metadata.py            # 数据集元信息
├── 📁 tests/                     # 测试代码
├── 📁 examples/                  # 使用示例
├── 📁 docs/                      # 项目文档
└── 📁 scripts/                   # 辅助脚本
```

## 🧪 测试系统

项目包含完整的自动化测试系统：

```bash
# 运行所有测试
python -m pytest tests/ -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html

# 使用便捷脚本
./run_tests.sh
```

## 🔧 开发环境

### Pre-commit配置

项目使用pre-commit确保代码质量：

```bash
# 安装pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push

# 手动运行检查
pre-commit run --all-files
```

### 代码质量工具

- **Ruff**: 快速Python代码检查和格式化
- **flake8**: 传统代码规范检查
- **mypy**: 静态类型检查
- **pytest**: 单元测试框架

## 📚 文档

- 📖 [开发指南](DEVELOPMENT.md) - 完整的开发环境设置和编码规范
- 🧪 [测试指南](TESTING.md) - 测试框架使用说明
- 📋 [API文档](docs/build/html/index.html) - 自动生成的API参考

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 开发流程

```bash
# 设置开发环境
conda activate pepbenchmark
export PYTHONPATH="${PYTHONPATH}:src"

# 运行测试
python -m pytest tests/ -v

# 代码质量检查
pre-commit run --all-files

# 生成文档
cd docs && make html
```

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 🔗 相关资源

- [项目主页](https://github.com/your-org/PepBenchmark)
- [问题报告](https://github.com/your-org/PepBenchmark/issues)
- [讨论区](https://github.com/your-org/PepBenchmark/discussions)

## 🙏 致谢

感谢所有为PepBenchmark项目贡献的开发者和研究人员！

---

> 💡 **提示**: 查看 [examples/](examples/) 目录获取更多使用示例



# 🧬 PepBenchmark 项目开发指南

## 📋 项目概述

PepBenchmark 是一个专门用于肽相关任务的综合性基准测试工具包，提供数据加载、特征提取、模型评估和可视化等功能。

## 🏗️ 项目结构

```
PepBenchmark/
├── 📁 src/pepbenchmark/           # 核心源代码
│   ├── 📁 dataset_loader/         # 数据集加载器
│   ├── 📁 pep_utils/             # 肽序列处理工具
│   ├── 📁 utils/                 # 通用工具函数
│   ├── 📁 visualization/         # 数据可视化
│   ├── 📄 metadata.py            # 数据集元信息
│   └── 📄 evaluator.py          # 模型评估器
├── 📁 tests/                     # 测试代码
├── 📁 examples/                  # 使用示例
├── 📁 docs/                      # 文档
├── 📁 scripts/                   # 辅助脚本
└── 📁 assets/                    # 资源文件
```

## 🚀 快速开始

### 环境设置

```bash
# 1. 克隆项目
git clone https://github.com/your-org/PepBenchmark.git
cd PepBenchmark

# 2. 创建conda环境
conda create -n pepbenchmark python=3.10
conda activate pepbenchmark

# 3. 安装依赖
pip install -e .
pip install -r requirements-test.txt

# 4. 设置pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

### 基础使用

```python
# 导入核心模块
from pepbenchmark.dataset_loader import SingleTaskDataset
from pepbenchmark.pep_utils import PeptideFeaturizer

# 加载数据集
dataset = SingleTaskDataset("BBP_APML")
data = dataset.get_data()

# 特征提取
featurizer = PeptideFeaturizer(
    input_format="fasta",
    feature_type="onehot"
)
features = featurizer.extract_features(data['sequence'])
```

## 🔧 开发环境配置

### Pre-commit 系统

项目使用完整的pre-commit系统确保代码质量：

#### 🛡️ 在每次提交时运行：
- ✅ **代码格式化**：自动修复空格、换行等格式问题
- ✅ **语法检查**：验证Python语法正确性
- ✅ **代码规范**：flake8 + ruff 双重检查
- ✅ **许可证头**：自动添加许可证信息
- ✅ **基础验证**：导入检查和基本功能验证

#### 🧪 在推送时运行：
- ✅ **完整测试套件**：运行所有单元测试
- ✅ **类型检查**：mypy静态类型检查
- ✅ **文档字符串**：pydoclint检查

### 代码质量工具

```bash
# 手动运行格式化
ruff format src/ tests/
ruff check src/ tests/ --fix

# 运行类型检查
mypy src/pepbenchmark --ignore-missing-imports

# 检查文档字符串
pydoclint src/pepbenchmark/
```

## 🧪 测试系统

### 测试结构

```
tests/
├── 📄 conftest.py                # 共享fixtures
├── 📄 test_metadata.py          # 元数据测试
├── 📁 test_dataset_loader/      # 数据加载测试
├── 📁 test_pep_utils/           # 肽工具测试
├── 📁 test_utils/               # 通用工具测试
└── 📁 fixtures/                 # 测试数据
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_dataset_loader/ -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html

# 运行快速测试（排除慢速测试）
python -m pytest tests/ -m "not slow" -v

# 使用便捷脚本
./run_tests.sh
```

### 测试分类

- 🏃‍♂️ **快速测试**：单元测试，使用mock，默认运行
- 🐌 **慢速测试**：集成测试，标记为`@pytest.mark.slow`
- 🧩 **集成测试**：跨模块测试，标记为`@pytest.mark.integration`

## 📝 编码规范

### 导入规范
```python
# ✅ 推荐：使用绝对导入
from pepbenchmark.dataset_loader.base_dataset import BaseDataset
from pepbenchmark.pep_utils.convert import Peptide

# ❌ 避免：相对导入在包外部使用
from .base_dataset import BaseDataset  # 仅在包内部使用
```

### 文档字符串
```python
def process_sequences(sequences: List[str], method: str = "default") -> pd.DataFrame:
    """
    处理肽序列数据。

    Args:
        sequences: 肽序列列表
        method: 处理方法，可选 "default", "advanced"

    Returns:
        处理后的DataFrame

    Raises:
        ValueError: 当method不被支持时

    Examples:
        >>> sequences = ["ALAG", "GGGC"]
        >>> result = process_sequences(sequences)
        >>> len(result) == 2
        True
    """
```

### 类型注解
```python
from typing import List, Dict, Optional, Union
import pandas as pd

def split_data(
    data: pd.DataFrame,
    fractions: List[float],
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """带有完整类型注解的函数示例。"""
    pass
```

## 🔍 调试指南

### 常见问题

#### 1. 导入错误
```bash
# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:src"

# 或在代码中
import sys
sys.path.insert(0, 'src')
```

#### 2. 测试失败
```bash
# 详细调试信息
python -m pytest tests/ -v -x --tb=long

# 进入调试器
python -m pytest tests/ --pdb

# 运行特定失败测试
python -m pytest tests/test_module.py::TestClass::test_method -v
```

#### 3. Pre-commit失败
```bash
# 跳过特定hook
git commit -m "message" --no-verify

# 手动运行特定hook
pre-commit run ruff --all-files
pre-commit run basic-validation --all-files

# 更新pre-commit hooks
pre-commit autoupdate
```

### 性能调试
```bash
# 使用pytest-benchmark
python -m pytest tests/ --benchmark-only

# 内存使用分析
python -m memory_profiler your_script.py

# 代码覆盖率分析
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

## 📚 文档系统

### 生成文档
```bash
# 进入文档目录
cd docs/

# 生成HTML文档
make html

# 清理并重新生成
make clean
make html

# 查看文档
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### API文档更新
- 文档使用Sphinx自动生成
- 修改源代码中的docstring会自动更新API文档
- 确保所有公共函数和类都有完整的docstring

## 🚢 发布流程

### 版本管理
```bash
# 1. 更新版本号
# 编辑 src/pepbenchmark/__init__.py 中的 __version__

# 2. 运行完整测试
./run_tests.sh

# 3. 生成文档
cd docs && make html

# 4. 创建标签
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### CI/CD集成
```yaml
# GitHub Actions 示例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-test.txt
    - name: Run tests
      run: python -m pytest tests/ --cov=src/pepbenchmark
```

## 🤝 贡献指南

### 提交代码流程

1. **创建分支**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **开发和测试**
   ```bash
   # 编写代码
   # 添加测试
   python -m pytest tests/ -v
   ```

3. **运行pre-commit检查**
   ```bash
   pre-commit run --all-files
   ```

4. **提交和推送**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/new-feature
   ```

5. **创建Pull Request**

### 代码审查清单

- [ ] 代码遵循项目编码规范
- [ ] 添加了相应的单元测试
- [ ] 测试覆盖率保持在合理水平
- [ ] 文档字符串完整准确
- [ ] Pre-commit检查全部通过
- [ ] 没有引入破坏性更改

## 📞 支持和帮助

### 获取帮助
- 📖 查看文档：`docs/build/html/index.html`
- 🧪 查看测试：`TESTING.md`
- 🔍 查看示例：`examples/` 目录
- 💬 提出Issue：GitHub Issues

### 常用命令速查

```bash
# 开发环境
conda activate pepbenchmark
export PYTHONPATH="${PYTHONPATH}:src"

# 代码质量
pre-commit run --all-files
ruff check src/ tests/ --fix
python -m pytest tests/ -v

# 文档生成
cd docs && make html

# 测试覆盖率
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html
```

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为PepBenchmark项目贡献代码、文档和想法的开发者！


# 🧬 PepBenchmark 项目开发指南

## 📋 项目概述

PepBenchmark 是一个专门用于肽相关任务的综合性基准测试工具包，提供数据加载、特征提取、模型评估和可视化等功能。

## 🏗️ 项目结构

```
PepBenchmark/
├── 📁 src/pepbenchmark/           # 核心源代码
│   ├── 📁 dataset_loader/         # 数据集加载器
│   ├── 📁 pep_utils/             # 肽序列处理工具
│   ├── 📁 utils/                 # 通用工具函数
│   ├── 📁 visualization/         # 数据可视化
│   ├── 📄 metadata.py            # 数据集元信息
│   └── 📄 evaluator.py          # 模型评估器
├── 📁 tests/                     # 测试代码
├── 📁 examples/                  # 使用示例
├── 📁 docs/                      # 文档
├── 📁 scripts/                   # 辅助脚本
└── 📁 assets/                    # 资源文件
```

## 🚀 快速开始

### 环境设置

```bash
# 1. 克隆项目
git clone https://github.com/your-org/PepBenchmark.git
cd PepBenchmark

# 2. 创建conda环境
conda create -n pepbenchmark python=3.10
conda activate pepbenchmark

# 3. 安装依赖
pip install -e .
pip install -r requirements-test.txt

# 4. 设置pre-commit hooks
pre-commit install
pre-commit install --hook-type pre-push
```

### 基础使用

```python
# 导入核心模块
from pepbenchmark.dataset_loader import SingleTaskDataset
from pepbenchmark.pep_utils import PeptideFeaturizer

# 加载数据集
dataset = SingleTaskDataset("BBP_APML")
data = dataset.get_data()

# 特征提取
featurizer = PeptideFeaturizer(
    input_format="fasta",
    feature_type="onehot"
)
features = featurizer.extract_features(data['sequence'])
```

## 🔧 开发环境配置

### Pre-commit 系统

项目使用完整的pre-commit系统确保代码质量：

#### 🛡️ 在每次提交时运行：
- ✅ **代码格式化**：自动修复空格、换行等格式问题
- ✅ **语法检查**：验证Python语法正确性
- ✅ **代码规范**：flake8 + ruff 双重检查
- ✅ **许可证头**：自动添加许可证信息
- ✅ **基础验证**：导入检查和基本功能验证

#### 🧪 在推送时运行：
- ✅ **完整测试套件**：运行所有单元测试
- ✅ **类型检查**：mypy静态类型检查
- ✅ **文档字符串**：pydoclint检查

### 代码质量工具

```bash
# 手动运行格式化
ruff format src/ tests/
ruff check src/ tests/ --fix

# 运行类型检查
mypy src/pepbenchmark --ignore-missing-imports

# 检查文档字符串
pydoclint src/pepbenchmark/
```

## 🧪 测试系统

### 测试结构

```
tests/
├── 📄 conftest.py                # 共享fixtures
├── 📄 test_metadata.py          # 元数据测试
├── 📁 test_dataset_loader/      # 数据加载测试
├── 📁 test_pep_utils/           # 肽工具测试
├── 📁 test_utils/               # 通用工具测试
└── 📁 fixtures/                 # 测试数据
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_dataset_loader/ -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html

# 运行快速测试（排除慢速测试）
python -m pytest tests/ -m "not slow" -v

# 使用便捷脚本
./run_tests.sh
```

### 测试分类

- 🏃‍♂️ **快速测试**：单元测试，使用mock，默认运行
- 🐌 **慢速测试**：集成测试，标记为`@pytest.mark.slow`
- 🧩 **集成测试**：跨模块测试，标记为`@pytest.mark.integration`

## 📝 编码规范

### 导入规范
```python
# ✅ 推荐：使用绝对导入
from pepbenchmark.dataset_loader.base_dataset import BaseDataset
from pepbenchmark.pep_utils.convert import Peptide

# ❌ 避免：相对导入在包外部使用
from .base_dataset import BaseDataset  # 仅在包内部使用
```

### 文档字符串
```python
def process_sequences(sequences: List[str], method: str = "default") -> pd.DataFrame:
    """
    处理肽序列数据。

    Args:
        sequences: 肽序列列表
        method: 处理方法，可选 "default", "advanced"

    Returns:
        处理后的DataFrame

    Raises:
        ValueError: 当method不被支持时

    Examples:
        >>> sequences = ["ALAG", "GGGC"]
        >>> result = process_sequences(sequences)
        >>> len(result) == 2
        True
    """
```

### 类型注解
```python
from typing import List, Dict, Optional, Union
import pandas as pd

def split_data(
    data: pd.DataFrame,
    fractions: List[float],
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """带有完整类型注解的函数示例。"""
    pass
```

## 🔍 调试指南

### 常见问题

#### 1. 导入错误
```bash
# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:src"

# 或在代码中
import sys
sys.path.insert(0, 'src')
```

#### 2. 测试失败
```bash
# 详细调试信息
python -m pytest tests/ -v -x --tb=long

# 进入调试器
python -m pytest tests/ --pdb

# 运行特定失败测试
python -m pytest tests/test_module.py::TestClass::test_method -v
```

#### 3. Pre-commit失败
```bash
# 跳过特定hook
git commit -m "message" --no-verify

# 手动运行特定hook
pre-commit run ruff --all-files
pre-commit run basic-validation --all-files

# 更新pre-commit hooks
pre-commit autoupdate
```

### 性能调试
```bash
# 使用pytest-benchmark
python -m pytest tests/ --benchmark-only

# 内存使用分析
python -m memory_profiler your_script.py

# 代码覆盖率分析
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

## 📚 文档系统

### 生成文档
```bash
# 进入文档目录
cd docs/

# 生成HTML文档
make html

# 清理并重新生成
make clean
make html

# 查看文档
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### API文档更新
- 文档使用Sphinx自动生成
- 修改源代码中的docstring会自动更新API文档
- 确保所有公共函数和类都有完整的docstring

## 🚢 发布流程

### 版本管理
```bash
# 1. 更新版本号
# 编辑 src/pepbenchmark/__init__.py 中的 __version__

# 2. 运行完整测试
./run_tests.sh

# 3. 生成文档
cd docs && make html

# 4. 创建标签
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### CI/CD集成
```yaml
# GitHub Actions 示例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -r requirements-test.txt
    - name: Run tests
      run: python -m pytest tests/ --cov=src/pepbenchmark
```

## 🤝 贡献指南

### 提交代码流程

1. **创建分支**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **开发和测试**
   ```bash
   # 编写代码
   # 添加测试
   python -m pytest tests/ -v
   ```

3. **运行pre-commit检查**
   ```bash
   pre-commit run --all-files
   ```

4. **提交和推送**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   git push origin feature/new-feature
   ```

5. **创建Pull Request**

### 代码审查清单

- [ ] 代码遵循项目编码规范
- [ ] 添加了相应的单元测试
- [ ] 测试覆盖率保持在合理水平
- [ ] 文档字符串完整准确
- [ ] Pre-commit检查全部通过
- [ ] 没有引入破坏性更改

## 📞 支持和帮助

### 获取帮助
- 📖 查看文档：`docs/build/html/index.html`
- 🧪 查看测试：`TESTING.md`
- 🔍 查看示例：`examples/` 目录
- 💬 提出Issue：GitHub Issues

### 常用命令速查

```bash
# 开发环境
conda activate pepbenchmark
export PYTHONPATH="${PYTHONPATH}:src"

# 代码质量
pre-commit run --all-files
ruff check src/ tests/ --fix
python -m pytest tests/ -v

# 文档生成
cd docs && make html

# 测试覆盖率
python -m pytest tests/ --cov=src/pepbenchmark --cov-report=html
```

---

## 📄 许可证

本项目采用 Apache License 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为PepBenchmark项目贡献代码、文档和想法的开发者！
