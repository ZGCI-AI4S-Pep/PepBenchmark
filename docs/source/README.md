# PepBenchmark 文档

欢迎来到 PepBenchmark 项目文档！本文档提供了完整的使用指南、API参考和示例代码。

## 文档结构

### 📚 用户指南
- **[项目介绍](introduction.md)** - 了解PepBenchmark的特性和应用
- **[安装指南](installation.md)** - 详细的安装和配置说明
- **[快速开始](quickstart.md)** - 通过示例快速上手
- **[用户指南](user_guide/index.md)** - 深入的功能说明和使用技巧

### 🗂️ 数据集
- **[数据集概览](datasets/overview.md)** - 所有可用数据集的详细信息
- **[构建数据集](construct_dataset.md)** - 如何创建自定义数据集

### 🔧 API 参考
- **[模块总览](api/modules.rst)** - 完整的API文档
- **[转换模块](api/convert.md)** - 分子表示格式转换
- **[评估模块](api/evaluator.md)** - 模型评估指标

### 💡 示例和教程
- **[肽性质预测](examples/property_prediction.md)** - 完整的预测流程示例

### 👥 开发者指南
- **[贡献指南](contributing.md)** - 如何为项目做贡献
- **[更改日志](changelog.md)** - 版本更新记录

## 快速导航

### 🚀 新用户开始
1. [安装PepBenchmark](installation.md)
2. [运行快速示例](quickstart.md)
3. [学习基本概念](user_guide/data_loading.md)

### 📊 数据科学家
1. [探索可用数据集](datasets/overview.md)
2. [了解分子表示方法](user_guide/molecular_representations.md)
3. [学习模型评估](user_guide/evaluation_metrics.md)

### 🔬 研究人员
1. [查看完整示例](examples/property_prediction.md)
2. [理解评估指标](api/evaluator.md)
3. [自定义数据集](construct_dataset.md)

### 💻 开发者
1. [阅读API文档](api/modules.rst)
2. [了解项目架构](contributing.md)
3. [参与开源贡献](contributing.md)

## 主要功能

### 🔄 格式转换
- FASTA ↔ SMILES ↔ HELM ↔ BiLN
- 分子指纹生成 (Morgan, MACCS, RDKit等)
- 神经网络嵌入 (ESM-2等预训练模型)

### 📈 模型评估
- 分类指标: Accuracy, F1, ROC-AUC, MCC等
- 回归指标: MAE, MSE, R², Pearson/Spearman相关
- 可视化工具: ROC曲线, 混淆矩阵, 特征重要性

### 🗃️ 数据管理
- 31个标准化肽数据集
- 多种数据分割策略
- 序列冗余去除和质量检查

### ⚡ 性能优化
- 批量处理和GPU加速
- 特征缓存机制
- 内存优化的大数据集处理

## 支持的任务

- **二分类**: 抗菌肽识别、细胞穿透性预测、毒性检测
- **回归**: 活性强度预测、物理化学性质计算
- **多分类**: 功能类别分类、作用机制预测

## 技术栈

- **Python 3.8+**
- **科学计算**: NumPy, Pandas, SciPy
- **机器学习**: Scikit-learn, PyTorch
- **化学信息**: RDKit, OGB
- **生物信息**: Biopython, HuggingFace Transformers
- **可视化**: Matplotlib, Seaborn, Plotly

## 获取帮助

### 📖 文档
- 搜索本文档获取详细信息
- 查看[API参考](api/modules.rst)了解函数用法
- 参考[示例代码](examples/)获取灵感

### 💬 社区支持
- [GitHub Issues](https://github.com/your-org/PepBenchmark/issues) - 报告问题和功能请求
- [GitHub Discussions](https://github.com/your-org/PepBenchmark/discussions) - 一般讨论和问题
- [邮件列表](mailto:pepbenchmark@example.com) - 获取更新和公告

### 🐛 问题排查
1. 检查[故障排除](user_guide/troubleshooting.md)页面
2. 搜索现有的GitHub Issues
3. 提供详细的错误信息和复现步骤

## 引用

如果您在研究中使用了PepBenchmark，请引用我们的工作：

```bibtex
@article{pepbenchmark2024,
  title={PepBenchmark: A Comprehensive Benchmark for Peptide Property Prediction},
  author={Your Team},
  journal={Bioinformatics},
  year={2024},
  doi={10.1093/bioinformatics/xxxx}
}
```

## 许可证

PepBenchmark 在 Apache License 2.0 下发布。详见 [LICENSE](https://github.com/your-org/PepBenchmark/blob/main/LICENSE) 文件。

## 致谢

感谢所有为PepBenchmark项目做出贡献的研究人员、开发者和用户。特别感谢：

- RDKit 团队提供的化学信息学工具
- HuggingFace 提供的预训练模型和工具
- 开源社区的持续支持和反馈

---

**文档版本**: v0.2.0
**最后更新**: 2024年1月
**语言**: 中文 | [English](README_en.md)
