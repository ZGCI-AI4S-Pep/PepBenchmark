# evaluator 模块 API 参考

`pepbenchmark.evaluator` 模块提供了全面的模型评估指标，支持二分类、多分类和回归任务。

## 概览

该模块实现了丰富的评估指标，包括：

- **分类指标**: 准确率、精确率、召回率、F1分数、AUC、MCC等
- **回归指标**: MAE、MSE、R²、Pearson和Spearman相关系数等
- **统计指标**: 特异性、Cohen's Kappa、Brier分数等

## 核心函数

### evaluate_classification

```{eval-rst}
.. autofunction:: pepbenchmark.evaluator.evaluate_classification

   评估分类模型的性能，支持多种评估指标。

   **参数:**

   - ``y_true (Union[List, np.ndarray])``: 真实标签
   - ``y_pred (Union[List, np.ndarray])``: 预测标签
   - ``y_score (Union[List, np.ndarray], optional)``: 预测概率或分数，用于概率相关指标
   - ``metrics (List[str], optional)``: 要计算的指标列表，None表示计算所有可用指标

   **返回:**

   - ``Dict[str, float]``: 指标名称及其计算值的字典

   **示例:**

   >>> y_true = [0, 1, 1, 0, 1]
   >>> y_pred = [0, 1, 0, 0, 1]
   >>> y_score = [0.1, 0.9, 0.4, 0.2, 0.8]
   >>>
   >>> results = evaluate_classification(y_true, y_pred, y_score)
   >>> print(f"Accuracy: {results['accuracy']:.3f}")
   >>> print(f"F1-Score: {results['f1']:.3f}")
   >>> print(f"ROC-AUC: {results['roc-auc']:.3f}")
```

### evaluate_regression

```{eval-rst}
.. autofunction:: pepbenchmark.evaluator.evaluate_regression

   评估回归模型的性能。

   **参数:**

   - ``y_true (Union[List, np.ndarray])``: 真实值
   - ``y_pred (Union[List, np.ndarray])``: 预测值
   - ``metrics (List[str], optional)``: 要计算的指标列表

   **返回:**

   - ``Dict[str, float]``: 指标名称及其计算值的字典

   **示例:**

   >>> y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
   >>> y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
   >>>
   >>> results = evaluate_regression(y_true, y_pred)
   >>> print(f"MAE: {results['mae']:.3f}")
   >>> print(f"R²: {results['r2']:.3f}")
   >>> print(f"Pearson: {results['pcc']:.3f}")
```

## 辅助函数

### specificity_score

```{eval-rst}
.. autofunction:: pepbenchmark.evaluator.specificity_score

   计算二分类任务的特异性(真负率)。

   **公式:**

   特异性 = TN / (TN + FP)

   其中 TN = 真负例数，FP = 假正例数

   **参数:**

   - ``y_true (Union[List, np.ndarray])``: 真实标签
   - ``y_pred (Union[List, np.ndarray])``: 预测标签

   **返回:**

   - ``float``: 特异性分数

   **示例:**

   >>> y_true = [0, 0, 1, 1, 0]
   >>> y_pred = [0, 1, 1, 1, 0]
   >>> spec = specificity_score(y_true, y_pred)
   >>> print(f"Specificity: {spec:.3f}")
```

### rmse_score

```{eval-rst}
.. autofunction:: pepbenchmark.evaluator.rmse_score

   计算均方根误差(Root Mean Square Error)。

   **公式:**

   RMSE = √(MSE) = √(Σ(y_true - y_pred)² / n)

   **参数:**

   - ``y_true (Union[List, np.ndarray])``: 真实值
   - ``y_pred (Union[List, np.ndarray])``: 预测值

   **返回:**

   - ``float``: RMSE值

   **示例:**

   >>> y_true = [1.0, 2.0, 3.0]
   >>> y_pred = [1.1, 1.9, 3.2]
   >>> rmse = rmse_score(y_true, y_pred)
   >>> print(f"RMSE: {rmse:.3f}")
```

## 评估器类

### BinaryClassificationEvaluator

```{eval-rst}
.. autoclass:: pepbenchmark.evaluator.BinaryClassificationEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   二分类任务的专用评估器。

   **主要方法:**

   - ``evaluate(y_true, y_pred, y_score=None)``: 执行全面评估
   - ``get_confusion_matrix(y_true, y_pred)``: 获取混淆矩阵
   - ``plot_roc_curve(y_true, y_score)``: 绘制ROC曲线
   - ``plot_precision_recall_curve(y_true, y_score)``: 绘制PR曲线

   **示例:**

   >>> evaluator = BinaryClassificationEvaluator()
   >>> metrics = evaluator.evaluate(y_true, y_pred, y_score)
   >>> cm = evaluator.get_confusion_matrix(y_true, y_pred)
```

### MultiClassificationEvaluator

```{eval-rst}
.. autoclass:: pepbenchmark.evaluator.MultiClassificationEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   多分类任务的专用评估器。

   **主要方法:**

   - ``evaluate(y_true, y_pred, y_score=None)``: 执行多分类评估
   - ``evaluate_per_class(y_true, y_pred)``: 按类别评估
   - ``plot_confusion_matrix(y_true, y_pred)``: 可视化混淆矩阵

   **示例:**

   >>> evaluator = MultiClassificationEvaluator()
   >>> metrics = evaluator.evaluate(y_true, y_pred)
   >>> per_class = evaluator.evaluate_per_class(y_true, y_pred)
```

### RegressionEvaluator

```{eval-rst}
.. autoclass:: pepbenchmark.evaluator.RegressionEvaluator
   :members:
   :undoc-members:
   :show-inheritance:

   回归任务的专用评估器。

   **主要方法:**

   - ``evaluate(y_true, y_pred)``: 执行回归评估
   - ``plot_scatter(y_true, y_pred)``: 绘制散点图
   - ``plot_residuals(y_true, y_pred)``: 绘制残差图

   **示例:**

   >>> evaluator = RegressionEvaluator()
   >>> metrics = evaluator.evaluate(y_true, y_pred)
   >>> evaluator.plot_scatter(y_true, y_pred)
```

## 支持的指标

### 分类指标

```{eval-rst}
.. list-table:: 支持的分类指标
   :header-rows: 1
   :widths: 20 30 50

   * - 指标名称
     - 函数名
     - 描述
   * - accuracy
     - accuracy_score
     - 分类准确率
   * - balanced-accuracy
     - balanced_accuracy_score
     - 平衡准确率
   * - precision
     - precision_score
     - 精确率
   * - recall
     - recall_score
     - 召回率(敏感性)
   * - specificity
     - specificity_score
     - 特异性
   * - f1
     - f1_score
     - F1分数
   * - mcc
     - matthews_corrcoef
     - 马修斯相关系数
   * - kappa
     - cohen_kappa_score
     - Cohen's Kappa系数
   * - roc-auc
     - roc_auc_score
     - ROC曲线下面积
   * - pr-auc
     - average_precision_score
     - PR曲线下面积
   * - brier-score
     - brier_score_loss
     - Brier分数
   * - log-loss
     - log_loss
     - 对数损失
```

### 回归指标

```{eval-rst}
.. list-table:: 支持的回归指标
   :header-rows: 1
   :widths: 20 30 50

   * - 指标名称
     - 函数名
     - 描述
   * - mse
     - mean_squared_error
     - 均方误差
   * - rmse
     - rmse_score
     - 均方根误差
   * - mae
     - mean_absolute_error
     - 平均绝对误差
   * - r2
     - r2_score
     - 决定系数
   * - pcc
     - pearsonr
     - Pearson相关系数
   * - spearman
     - spearmanr
     - Spearman等级相关系数
```

## 使用示例

### 基本分类评估

```python
from pepbenchmark.evaluator import evaluate_classification
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 准备数据
X, y = load_data()  # 你的数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]  # 正类概率

# 评估
metrics = evaluate_classification(y_test, y_pred, y_score)

print("分类评估结果:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### 基本回归评估

```python
from pepbenchmark.evaluator import evaluate_regression
from sklearn.ensemble import RandomForestRegressor

# 准备数据
X, y = load_regression_data()  # 你的回归数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
metrics = evaluate_regression(y_test, y_pred)

print("回归评估结果:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### 使用评估器类

```python
from pepbenchmark.evaluator import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

# 创建评估器
evaluator = BinaryClassificationEvaluator()

# 执行评估
metrics = evaluator.evaluate(y_test, y_pred, y_score)
print("详细评估结果:", metrics)

# 获取混淆矩阵
cm = evaluator.get_confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 绘制ROC曲线
evaluator.plot_roc_curve(y_test, y_score)
plt.title("ROC Curve")
plt.show()

# 绘制Precision-Recall曲线
evaluator.plot_precision_recall_curve(y_test, y_score)
plt.title("Precision-Recall Curve")
plt.show()
```

### 自定义指标

```python
def custom_metric(y_true, y_pred):
    """自定义评估指标"""
    # 你的自定义计算逻辑
    return np.mean((y_true - y_pred) ** 2)

# 添加到现有评估中
metrics = evaluate_classification(y_test, y_pred, y_score)
metrics['custom'] = custom_metric(y_test, y_pred)
```

### 批量模型比较

```python
from pepbenchmark.evaluator import ModelComparator

# 定义多个模型
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'LogisticRegression': LogisticRegression()
}

# 创建比较器
comparator = ModelComparator(models)

# 执行比较
results = comparator.compare(X_train, y_train, X_test, y_test,
                           metrics=['accuracy', 'f1', 'roc-auc'])

print("模型比较结果:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

### 交叉验证评估

```python
from pepbenchmark.evaluator import CrossValidationEvaluator
from sklearn.model_selection import StratifiedKFold

# 创建交叉验证评估器
cv_evaluator = CrossValidationEvaluator(
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# 执行交叉验证评估
cv_results = cv_evaluator.evaluate(model, X, y,
                                 metrics=['accuracy', 'f1', 'roc-auc'])

print("交叉验证结果:")
for metric, scores in cv_results.items():
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")
```

## 性能考虑

### 大数据集处理

对于大型数据集，建议分批计算指标：

```python
def evaluate_large_dataset(y_true, y_pred, y_score=None, batch_size=10000):
    """分批评估大数据集"""
    n_samples = len(y_true)

    # 对于某些指标，需要全量数据
    global_metrics = ['roc-auc', 'pr-auc']

    # 可以分批计算的指标
    batch_metrics = ['accuracy', 'precision', 'recall', 'f1']

    results = {}

    # 全局指标
    for metric in global_metrics:
        if y_score is not None:
            results[metric] = evaluate_classification(
                y_true, y_pred, y_score, metrics=[metric]
            )[metric]

    # 分批指标
    batch_results = {metric: [] for metric in batch_metrics}

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_y_true = y_true[i:end_idx]
        batch_y_pred = y_pred[i:end_idx]

        batch_metrics_results = evaluate_classification(
            batch_y_true, batch_y_pred, metrics=batch_metrics
        )

        for metric in batch_metrics:
            batch_results[metric].append(batch_metrics_results[metric])

    # 计算平均值
    for metric in batch_metrics:
        results[metric] = np.mean(batch_results[metric])

    return results
```

### 内存优化

```python
import gc

def memory_efficient_evaluation(model, X_test, y_test, batch_size=1000):
    """内存优化的模型评估"""
    y_pred = []
    y_score = []

    # 分批预测
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i+batch_size]

        batch_pred = model.predict(batch_X)
        batch_score = model.predict_proba(batch_X)[:, 1]

        y_pred.extend(batch_pred)
        y_score.extend(batch_score)

        # 释放内存
        del batch_X, batch_pred, batch_score
        gc.collect()

    # 评估
    return evaluate_classification(y_test, y_pred, y_score)
```

## 最佳实践

1. **选择合适的指标**: 根据任务类型和业务需求选择评估指标
2. **处理类别不平衡**: 使用平衡准确率、F1分数、MCC等指标
3. **概率校准**: 对于需要概率输出的模型，考虑概率校准
4. **统计显著性**: 进行多次评估并报告置信区间
5. **可视化结果**: 使用ROC、PR曲线等可视化工具辅助分析

## 相关模块

- [convert](convert.md) - 分子表示转换
- [splitter](splitter.md) - 数据分割策略
- [visualization](../user_guide/visualization.md) - 结果可视化
