# 数据集概览

PepBenchmark 提供了丰富的标准化肽数据集，涵盖多种生物活性和应用场景。所有数据集都经过精心整理和验证，确保数据质量和一致性。

## 数据集分类

### 按任务类型分类

- **二分类任务**: 判断肽是否具有特定性质(如抗菌活性、细胞穿透性等)
- **回归任务**: 预测肽的连续数值属性(如活性强度、溶解度等)

### 按分子类型分类

- **天然肽**: 仅包含20种标准氨基酸的肽序列
- **合成肽**: 包含修饰氨基酸、非天然氨基酸或其他化学修饰的肽

### 按应用场景分类：

- **ADME**:
- **Theraputic-AMP**:
- ...


## 数据集统计



## 天然肽二分类数据集

这些数据集包含仅由20种标准氨基酸组成的肽序列，用于二分类任务。

### 抗菌肽相关
- **AMP_PepDiffusion**: 抗菌肽识别和分类
- **AMP_GRAMPA**: 革兰氏阳性/阴性特异性抗菌肽
- **AF_APML**: 抗真菌肽预测
- **AP_APML**: 抗寄生虫肽识别
- **AV_APML**: 抗病毒肽预测

### 细胞穿透肽
- **cCPP_Pepland**: 阳离子细胞穿透肽
- **BBP_APML**: 血脑屏障穿透肽

### 毒性和安全性
- **Hemo_PeptideBERT**: 溶血肽预测
- **Tox_APML**: 一般毒性肽识别

### 物理化学性质
- **Solubility**: 肽溶解度预测

### 特殊功能
- **QS_APML**: 群体感应肽
- **PepPI**: 蛋白质-蛋白质相互作用肽


## 合成肽二分类数据集

包含修饰氨基酸、环状结构或其他化学修饰的肽序列。

### 环肽相关
- **ncCPP_CycPeptMPDB**: 非阳离子环肽细胞穿透性
- **CycPeptMPDB_PAMA**: 环肽膜活性预测

### 特殊功能
- **Nonfouling**: 防污肽性能预测
- **cAB_APML2**: 阳离子抗菌肽(含修饰)
- **ncAB_APML2**: 非阳离子抗菌肽(含修饰)


## 天然肽回归数据集

用于预测天然肽的连续数值属性。

### 抗菌活性
- **AMP-MIC**: 最小抑菌浓度预测

### 酶抑制活性
- **ACE_APML**: ACE抑制肽活性
- **DPPIV_APML**: DPP-IV抑制肽活性
- **Aox_APML**: 抗氧化肽活性

### 神经活性
- **Neuro_APML**: 神经肽活性预测

### 其他生物活性
- **ACP_APML**: 抗癌肽活性
- **DLAD_BioDADPep**: 特定生物活性预测


## 合成肽回归数据集

包含修饰肽的活性预测任务。

### 代谢相关
- **TTCA_TCAHybrid**: 三羧酸循环相关活性

### 特殊修饰肽
- **ncAV_APML2**: 非阳离子抗病毒肽活性

[详细信息 →](synthetic_regression.md)

## 数据集特点

### 数据质量保证
- **去重处理**: 所有数据集都经过严格的重复序列去除
- **质量验证**: 序列格式和标签一致性验证
- **标准化**: 统一的数据格式和命名规范
- **文档完善**: 每个数据集都有详细的来源和处理说明

### 数据分布特征

```python
# 查看数据集的基本统计信息
from pepbenchmark.metadata import get_dataset_statistics

stats = get_dataset_statistics()
print("数据集统计信息:")
for category, datasets in stats.items():
    print(f"\n{category}:")
    for dataset_key, info in datasets.items():
        print(f"  {dataset_key}: {info['num_samples']} 样本, "
              f"平均长度 {info['avg_length']:.1f}, "
              f"标准差 {info['length_std']:.1f}")
```


## 数据访问和使用

### 快速加载

```python
from pepbenchmark.single_peptide.singeltask_dataset import SingleTaskDatasetManager

# 加载任意数据集
dataset = SingleTaskDatasetManager(dataset_name="BBP_APML", official_feature_names=["fasta", "label"])
print(f"数据集大小: {len(dataset)}")

# 获取序列和标签
sequences = dataset.get_official_feature("fasta")
labels = dataset.get_official_feature("label")
```

### 批量分析

```python
from pepbenchmark.metadata import natural_binary_keys
from pepbenchmark.utils.analysis import DatasetAnalyzer

# 分析所有天然肽二分类数据集
analyzer = DatasetAnalyzer()
for dataset_key in natural_binary_keys:
    analysis = analyzer.analyze(dataset_key)
    print(f"{dataset_key}: {analysis.summary()}")
```


### 贡献指南
我们欢迎社区贡献新的数据集或改进现有数据集。请参考[贡献指南](../contributing.md)了解如何参与。

## 引用和致谢

使用 PepBenchmark 数据集时，请引用相应的原始数据来源和本项目：

```bibtex
@article{pepbenchmark2024,
  title={PepBenchmark: A Comprehensive Benchmark for Peptide},
  author={Your Team},
  journal={Journal Name},
  year={2024}
}
```

每个数据集页面都包含具体的引用信息，请确保正确引用原始数据来源。

## 下一步
- 查看具体数据集类别的详细信息
- 学习[官方数据加载](../user_guide/data_loading.md)技术
- 了解如何[构建自定义数据集](../construct_dataset.md)
- 探索[示例应用](../examples/)
