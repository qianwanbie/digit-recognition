# 📊 实验说明 / Experiments Overview

---

## 1️⃣ 实验结果记录 / Experiment Results

| 实验编号 / Experiment ID | 模型 / Model | 数据版本 / Data Version | 超参数 / Hyperparameters | 精度 / Accuracy | 备注 / Notes |
|--------------------------|------------|-----------------------|------------------------|----------------|---------------|
| exp1 | RandomForest | v2 | n_estimators=100, max_depth=10 | 0.92 | 初始训练实验 / Initial training experiment |
| exp2 | RandomForest | v2 | n_estimators=200, max_depth=15 | 0.94 | 增加树数和深度 / Increased number of trees and depth |
| exp3 | RandomForest | v2 | n_estimators=300, max_depth=15 | 0.945 | 调整超参数优化指标 / Tuned hyperparameters for better accuracy |

> 可根据 MLflow 或训练日志补充更多实验记录  
> Add more experiment entries based on MLflow logs or training records.

---

## 2️⃣ 可用于生产的实验及原因 / Production-ready Experiment and Reason

| 实验编号 / Experiment ID | 原因 / Reason |
|--------------------------|---------------|
| exp3 | 在验证集和测试集上均表现最好，精度最高且训练稳定，适合作为生产模型 / Achieved the best performance on validation and test sets, with highest accuracy and stable training; suitable as production model |

> 选择生产模型时应考虑指标表现、稳定性和资源消耗  
> When selecting a production model, consider performance metrics, stability, and resource consumption.

---

## 3️⃣ 优化指标及重要性 / Optimized Metrics and Importance

| 指标 / Metric | 优化目标 / Optimization Goal | 重要性 / Importance |
|---------------|----------------------------|-------------------|
| Accuracy / 准确率 | 最大化 / Maximize | 核心指标，用于衡量模型整体性能 / Core metric to measure overall model performance |
| Confusion Matrix / 混淆矩阵 | 减少分类错误 / Reduce misclassification | 帮助分析不同类别的预测误差 / Helps analyze prediction errors per class |
| Training Time / 训练时间 | 尽量缩短 / Minimize | 平衡性能和效率，适合生产部署 / Balance performance and efficiency for production deployment |
