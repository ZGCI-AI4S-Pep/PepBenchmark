# 数据预处理文档

本目录包含了PepBenchmark数据预处理功能的完整文档。

## 文档结构

### 主要文档

1. **[数据预处理指南](./preprocessing.md)** - 完整的数据预处理流程
   - 分类数据集预处理
   - 回归数据集预处理
   - 特征生成
   - 数据划分
   - 参数说明和示例

### 核心类文档

2. **[Redundancy类文档](./redundancy.md)** - 冗余去除功能
   - 相似性分析
   - 多种去重方法（MMseqs2、CD-HIT、AAR）
   - 可视化功能
   - 性能优化建议

3. **[NegSampler类文档](./neg_sampler.md)** - 负样本采样功能
   - 智能负样本采样策略
   - 属性匹配采样
   - 多种采样池支持
   - 可视化分析

4. **[特征转换器文档](./converter.md)** - 特征转换功能
   - FASTA序列转换器
   - SMILES转换器
   - 嵌入生成
   - 分子指纹和图表示

### 相关文档

5. **[数据划分文档](./data_splitting.md)** - 数据划分策略
   - 随机划分
   - MMseqs2同源感知划分
   - CD-HIT划分
   - K折交叉验证

## 快速开始

### 分类数据集预处理

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

### 回归数据集预处理

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

### 特征生成

```python
from pepbenchmark.preprocess import preprocess_dataset

# 生成所有支持的特征
preprocessor = preprocess_dataset(
    dataset_name="BBP",
    feature_types=None,  # 生成所有特征
    save_results=True
)

# 生成特定特征
preprocessor = preprocess_dataset(
    dataset_name="BBP",
    feature_types=["fasta", "smiles", "ecfp4", "esm2_150_embedding"],
    save_results=True
)
```

## 支持的特征类型

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

## 支持的数据划分策略

- **random_split**: 随机划分
- **mmseqs2_split**: MMseqs2同源感知划分
- **cdhit_split**: CD-HIT划分

## 支持的去重方法

- **mmseqs**: MMseqs2快速去重
- **cdhit**: CD-HIT精确去重
- **aar**: 基于氨基酸残差的快速去重

## 支持的异常值检测方法

- **iqr**: 基于四分位距的异常值检测
- **zscore**: 基于Z-score的异常值检测

## 工作流程

### 分类数据集工作流程

1. **数据加载**: 加载正样本序列
2. **长度过滤**: 过滤过长序列
3. **冗余去除**: 去除相似序列
4. **负样本采样**: 生成负样本
5. **数据组合**: 组合正负样本
6. **数据验证**: 验证数据质量
7. **特征生成**: 生成多种特征
8. **数据划分**: 划分训练/验证/测试集

### 回归数据集工作流程

1. **数据加载**: 加载序列和标签
2. **长度过滤**: 过滤过长序列
3. **异常值检测**: 检测和移除异常值
4. **数据去重**: 去除重复序列
5. **数据验证**: 验证数据质量
6. **特征生成**: 生成多种特征
7. **数据划分**: 划分训练/验证/测试集

## 性能优化建议

### 并行处理

```python
# 使用多进程加速
processes=8  # 根据CPU核心数调整
```

### 缓存机制

- Redundancy类：自动缓存相似性计算结果
- NegSampler类：缓存采样池和属性计算结果
- 特征转换器：缓存转换结果

### 内存优化

- 分批处理大规模数据集
- 使用适当的批处理大小
- 及时释放不需要的变量

## 错误处理

### 常见错误及解决方案

1. **依赖工具未安装**
   - MMseqs2: `sudo apt-get install mmseqs2`
   - CD-HIT: `pip install cd-hit`

2. **内存不足**
   - 减少并行进程数
   - 减少批处理大小
   - 分批处理数据

3. **模型下载失败**
   - 检查网络连接
   - 使用本地模型路径

## 相关资源

- [PepBenchmark GitHub](https://github.com/ZGCI-AI4S-Pep/PepBenchmark)
- [官方数据集文档](./official_dataset.md)
- [用户自定义数据集文档](./usert_dataset.md)
- [模型训练文档](./model_training.md)
- [评估指标文档](./evaluation_metrics.md)
- [可视化文档](./visualization.md)
