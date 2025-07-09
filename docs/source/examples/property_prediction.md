# 肽性质预测示例

本示例演示如何使用 PepBenchmark 进行完整的肽性质预测任务，包括数据加载、特征工程、模型训练和评估。

## 示例概述

我们将使用血脑屏障穿透肽(BBP_APML)数据集，演示一个完整的二分类预测流程：

1. 数据加载和探索
2. 特征生成和选择
3. 数据分割和预处理
4. 模型训练和优化
5. 模型评估和可视化
6. 结果分析和解释

## 完整代码示例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

# PepBenchmark imports
from pepbenchmark.single_pred.dataset import PeptideDataset
from pepbenchmark.pep_utils.convert import Fasta2Smiles, Smiles2FP, Fasta2Embedding
from pepbenchmark.evaluator import BinaryClassificationEvaluator
from pepbenchmark.splitter import RandomSplitter, ClusterSplitter
from pepbenchmark.visualization import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_sequence_length_distribution
)

# 设置随机种子
np.random.seed(42)

print("🧬 PepBenchmark 肽性质预测示例")
print("=" * 50)
```

### 步骤 1: 数据加载和探索

```python
# 1.1 加载数据集
print("📊 步骤 1: 数据加载和探索")
print("-" * 30)

dataset_key = "BBP_APML"  # 血脑屏障穿透肽
dataset = PeptideDataset(dataset_key)

print(f"数据集: {dataset_key}")
print(f"样本数量: {len(dataset)}")
print(f"可用特征: {dataset.get_feature_names()}")

# 1.2 获取基本数据
sequences = dataset.get_sequences()
labels = dataset.get_labels()

print(f"序列数量: {len(sequences)}")
print(f"标签分布: {np.bincount(labels)}")
print(f"正样本比例: {labels.mean():.3f}")

# 1.3 转换为DataFrame进行分析
df = dataset.to_dataframe()
df['sequence_length'] = df['sequence'].str.len()

print(f"\n序列长度统计:")
print(df['sequence_length'].describe())

# 1.4 可视化序列长度分布
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plot_sequence_length_distribution(sequences, title="序列长度分布")

plt.subplot(1, 2, 2)
df.groupby('label')['sequence_length'].hist(alpha=0.7, bins=20)
plt.xlabel('序列长度')
plt.ylabel('频次')
plt.title('按类别的序列长度分布')
plt.legend(['负样本', '正样本'])

plt.tight_layout()
plt.show()

# 1.5 显示样本示例
print(f"\n样本示例:")
for i in range(3):
    print(f"  序列 {i+1}: {sequences[i][:30]}... (长度: {len(sequences[i])}, 标签: {labels[i]})")
```

### 步骤 2: 特征工程

```python
print(f"\n🔧 步骤 2: 特征工程")
print("-" * 30)

# 2.1 生成多种分子表示
feature_types = []
feature_arrays = []
feature_names = []

# 2.1.1 Morgan指纹
print("生成 Morgan 指纹...")
try:
    morgan_features = dataset.get_features("Morgan", radius=2, nBits=2048)
    feature_types.append("Morgan")
    feature_arrays.append(morgan_features)
    feature_names.append("Morgan_2048")
    print(f"  Morgan指纹形状: {morgan_features.shape}")
except Exception as e:
    print(f"  Morgan指纹生成失败: {e}")

# 2.1.2 MACCS指纹
print("生成 MACCS 指纹...")
try:
    maccs_features = dataset.get_features("MACCS")
    feature_types.append("MACCS")
    feature_arrays.append(maccs_features)
    feature_names.append("MACCS_167")
    print(f"  MACCS指纹形状: {maccs_features.shape}")
except Exception as e:
    print(f"  MACCS指纹生成失败: {e}")

# 2.1.3 ESM-2嵌入 (如果GPU可用)
print("生成 ESM-2 嵌入...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")

    embedder = Fasta2Embedding("facebook/esm2_t30_150M_UR50D", device=device)
    # 分批处理以避免内存问题
    batch_size = 32
    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_embs = embedder(batch_seqs)
        embeddings.extend(batch_embs)

    embedding_features = np.array(embeddings)
    feature_types.append("ESM2")
    feature_arrays.append(embedding_features)
    feature_names.append("ESM2_640")
    print(f"  ESM-2嵌入形状: {embedding_features.shape}")

except Exception as e:
    print(f"  ESM-2嵌入生成失败: {e}")

# 2.1.4 氨基酸组成特征
print("生成氨基酸组成特征...")
try:
    aac_features = dataset.get_features("AAC")
    feature_types.append("AAC")
    feature_arrays.append(aac_features)
    feature_names.append("AAC_20")
    print(f"  氨基酸组成特征形状: {aac_features.shape}")
except Exception as e:
    print(f"  氨基酸组成特征生成失败: {e}")

# 2.2 特征统计
print(f"\n特征生成完成，共 {len(feature_arrays)} 种特征类型:")
for i, (ftype, farray, fname) in enumerate(zip(feature_types, feature_arrays, feature_names)):
    sparsity = (farray == 0).mean()
    print(f"  {i+1}. {fname}: {farray.shape}, 稀疏度: {sparsity:.3f}")
```

### 步骤 3: 数据分割

```python
print(f"\n📊 步骤 3: 数据分割")
print("-" * 30)

# 3.1 随机分割
print("执行随机分割...")
random_splitter = RandomSplitter(test_size=0.2, random_state=42)

# 使用第一个可用的特征进行分割
if feature_arrays:
    X = feature_arrays[0]  # 使用Morgan指纹作为主要特征
    y = labels

    train_idx, test_idx = random_splitter.split(X, y)

    print(f"训练集大小: {len(train_idx)}")
    print(f"测试集大小: {len(test_idx)}")
    print(f"训练集正样本比例: {y[train_idx].mean():.3f}")
    print(f"测试集正样本比例: {y[test_idx].mean():.3f}")

# 3.2 聚类分割 (可选)
print("\n执行聚类分割 (基于序列相似性)...")
try:
    cluster_splitter = ClusterSplitter(n_clusters=5, test_size=0.2, random_state=42)
    train_idx_cluster, test_idx_cluster = cluster_splitter.split(X, y)

    print(f"聚类分割 - 训练集: {len(train_idx_cluster)}, 测试集: {len(test_idx_cluster)}")
    print(f"聚类分割 - 训练集正样本比例: {y[train_idx_cluster].mean():.3f}")
    print(f"聚类分割 - 测试集正样本比例: {y[test_idx_cluster].mean():.3f}")

    # 使用聚类分割
    train_idx, test_idx = train_idx_cluster, test_idx_cluster

except Exception as e:
    print(f"聚类分割失败，使用随机分割: {e}")
```

### 步骤 4: 模型训练和优化

```python
print(f"\n🤖 步骤 4: 模型训练和优化")
print("-" * 30)

# 4.1 准备训练数据
models_results = {}

# 对每种特征类型训练模型
for feat_idx, (ftype, X_feat, fname) in enumerate(zip(feature_types, feature_arrays, feature_names)):
    print(f"\n训练特征类型: {fname}")
    print("-" * 20)

    # 分割数据
    X_train, X_test = X_feat[train_idx], X_feat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4.2 定义模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    feature_results = {}

    # 4.3 训练每个模型
    for model_name, model in models.items():
        print(f"  训练 {model_name}...")

        # 简单训练
        if model_name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_score = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]

        # 评估
        evaluator = BinaryClassificationEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred, y_score)

        feature_results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_score': y_score,
            'model': model
        }

        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        print(f"    ROC-AUC: {metrics['roc-auc']:.4f}")

    models_results[fname] = feature_results

# 4.4 模型优化 (网格搜索)
print(f"\n🔍 模型优化 - 网格搜索")
print("-" * 20)

# 选择最佳特征类型进行优化
best_feature = feature_names[0]  # 使用第一个特征作为示例
X_best = feature_arrays[0]
X_train, X_test = X_best[train_idx], X_best[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# RandomForest网格搜索
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

print("执行 RandomForest 网格搜索...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print(f"最佳RF参数: {rf_grid.best_params_}")
print(f"最佳CV得分: {rf_grid.best_score_:.4f}")

# 使用最佳模型预测
y_pred_best = best_rf.predict(X_test)
y_score_best = best_rf.predict_proba(X_test)[:, 1]
```

### 步骤 5: 详细评估

```python
print(f"\n📈 步骤 5: 详细评估")
print("-" * 30)

# 5.1 全面评估最佳模型
evaluator = BinaryClassificationEvaluator()
final_metrics = evaluator.evaluate(y_test, y_pred_best, y_score_best)

print("最佳模型评估结果:")
for metric, value in final_metrics.items():
    print(f"  {metric}: {value:.4f}")

# 5.2 混淆矩阵
cm = evaluator.get_confusion_matrix(y_test, y_pred_best)
print(f"\n混淆矩阵:")
print(cm)

# 5.3 分类报告
print(f"\n分类报告:")
print(classification_report(y_test, y_pred_best, target_names=['非穿透', '穿透']))

# 5.4 可视化评估结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 混淆矩阵
plot_confusion_matrix(y_test, y_pred_best, ax=axes[0,0],
                     title="混淆矩阵")

# ROC曲线
plot_roc_curve(y_test, y_score_best, ax=axes[0,1],
               title="ROC曲线")

# 特征重要性
if hasattr(best_rf, 'feature_importances_'):
    plot_feature_importance(best_rf.feature_importances_,
                          ax=axes[1,0], title="特征重要性 (Top 20)")

# 预测分布
axes[1,1].hist(y_score_best[y_test==0], alpha=0.7, bins=20, label='负样本')
axes[1,1].hist(y_score_best[y_test==1], alpha=0.7, bins=20, label='正样本')
axes[1,1].set_xlabel('预测概率')
axes[1,1].set_ylabel('频次')
axes[1,1].set_title('预测概率分布')
axes[1,1].legend()

plt.tight_layout()
plt.show()
```

### 步骤 6: 结果分析

```python
print(f"\n🔍 步骤 6: 结果分析")
print("-" * 30)

# 6.1 比较不同特征类型的性能
print("不同特征类型性能比较:")
print("-" * 40)

performance_summary = []
for fname, feat_results in models_results.items():
    for model_name, result in feat_results.items():
        metrics = result['metrics']
        performance_summary.append({
            'Feature': fname,
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1': metrics['f1'],
            'ROC-AUC': metrics['roc-auc'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })

perf_df = pd.DataFrame(performance_summary)
print(perf_df.round(4))

# 6.2 绘制性能比较图
plt.figure(figsize=(15, 10))

# 准确率比较
plt.subplot(2, 2, 1)
pivot_acc = perf_df.pivot(index='Feature', columns='Model', values='Accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='viridis')
plt.title('准确率比较')

# F1分数比较
plt.subplot(2, 2, 2)
pivot_f1 = perf_df.pivot(index='Feature', columns='Model', values='F1')
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='viridis')
plt.title('F1分数比较')

# ROC-AUC比较
plt.subplot(2, 2, 3)
pivot_auc = perf_df.pivot(index='Feature', columns='Model', values='ROC-AUC')
sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='viridis')
plt.title('ROC-AUC比较')

# 综合性能雷达图
plt.subplot(2, 2, 4)
best_combo = perf_df.loc[perf_df['ROC-AUC'].idxmax()]
metrics_names = ['Accuracy', 'F1', 'ROC-AUC', 'Precision', 'Recall']
values = [best_combo[m] for m in metrics_names]

angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False)
values_plot = values + [values[0]]  # 闭合图形
angles_plot = np.concatenate([angles, [angles[0]]])

ax = plt.subplot(2, 2, 4, projection='polar')
ax.plot(angles_plot, values_plot, 'bo-', linewidth=2)
ax.fill(angles_plot, values_plot, alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1)
ax.set_title(f'最佳模型性能\n({best_combo["Feature"]} + {best_combo["Model"]})')

plt.tight_layout()
plt.show()

# 6.3 错误分析
print(f"\n错误分析:")
print("-" * 20)

# 找出预测错误的样本
misclassified_idx = test_idx[y_test != y_pred_best]
misclassified_seqs = [sequences[i] for i in misclassified_idx]
misclassified_true = y_test[y_test != y_pred_best]
misclassified_pred = y_pred_best[y_test != y_pred_best]
misclassified_score = y_score_best[y_test != y_pred_best]

print(f"错误分类样本数: {len(misclassified_idx)}")
print(f"错误率: {len(misclassified_idx) / len(y_test):.3f}")

# 显示一些错误分类的示例
print(f"\n错误分类示例:")
for i in range(min(5, len(misclassified_seqs))):
    seq = misclassified_seqs[i]
    true_label = misclassified_true[i]
    pred_label = misclassified_pred[i]
    score = misclassified_score[i]

    print(f"  序列: {seq[:40]}...")
    print(f"  真实标签: {true_label}, 预测标签: {pred_label}, 预测概率: {score:.3f}")
    print()

# 6.4 序列长度对预测的影响
print("序列长度对预测准确性的影响:")
test_sequences = [sequences[i] for i in test_idx]
test_lengths = [len(seq) for seq in test_sequences]

# 按长度分组分析
length_bins = [0, 10, 20, 30, 50, 100]
for i in range(len(length_bins) - 1):
    min_len, max_len = length_bins[i], length_bins[i+1]
    mask = (np.array(test_lengths) >= min_len) & (np.array(test_lengths) < max_len)

    if mask.sum() > 0:
        bin_accuracy = (y_test[mask] == y_pred_best[mask]).mean()
        print(f"  长度 {min_len}-{max_len}: {mask.sum()} 样本, 准确率: {bin_accuracy:.3f}")

print(f"\n🎉 预测任务完成!")
print(f"最佳性能: {best_combo['Feature']} + {best_combo['Model']}")
print(f"最终ROC-AUC: {best_combo['ROC-AUC']:.4f}")
```

## 代码要点解释

### 1. 特征工程策略
- **多种表示方法**: 结合分子指纹、序列嵌入、组成特征
- **批处理**: 大数据集分批处理避免内存问题
- **错误处理**: 优雅处理特征生成失败的情况

### 2. 模型选择和评估
- **多模型比较**: RandomForest、SVM、LogisticRegression
- **网格搜索**: 自动寻找最佳超参数
- **全面评估**: 使用多种指标评估模型性能

### 3. 可视化分析
- **性能对比**: 热力图比较不同组合的性能
- **错误分析**: 分析预测错误的样本特征
- **雷达图**: 综合展示最佳模型的各项指标

### 4. 生产级考虑
- **随机种子**: 确保结果可重现
- **内存管理**: 处理大数据集的内存优化
- **模块化**: 代码结构清晰，易于维护和扩展

## 扩展建议

1. **集成学习**: 尝试组合多种特征类型的预测结果
2. **深度学习**: 使用神经网络模型进行端到端学习
3. **特征选择**: 实施特征重要性分析和选择算法
4. **交叉验证**: 使用更严格的交叉验证策略
5. **超参数优化**: 使用贝叶斯优化等高级方法

这个示例提供了一个完整的肽性质预测工作流程，可以作为实际项目的起点。
