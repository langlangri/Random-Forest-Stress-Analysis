import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from itertools import cycle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 加载数据
file_path = 'StressLevelDataset.csv'
data = pd.read_csv(file_path)

# 查看数据结构，确保正确加载
print(data.head())

# 划分特征和目标变量
X = data.drop('stress_level', axis=1)
y = data['stress_level']

# 样本数量划分70% 训练集, 10% 验证集, 20% 测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# 模型选择与训练
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("找到的最佳参数: ", grid_search.best_params_)

# 在验证集评估模型-最佳参数
val_predictions = best_model.predict(X_val)
print("验证集分类报告:")
print(classification_report(y_val, val_predictions))

# 在测试集上评估模型
test_predictions = best_model.predict(X_test)
print("测试集分类报告:")
print(classification_report(y_test, test_predictions))

# 混淆矩阵
cm = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=True, square=True,
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('测试集混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 准确率
accuracy = accuracy_score(y_test, test_predictions)
print(f"测试集上的准确率: {accuracy:.4f}")

# 特征重要性
feature_importances = best_model.feature_importances_
sorted_idx = feature_importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.xticks(range(len(sorted_idx)), X.columns[sorted_idx], rotation=90)
plt.title('压力水平的特征重要性')
plt.show()

# ROC曲线图
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# 计算预测概率
y_score = best_model.predict_proba(X_test)

# 计算每一类的FPR和TPR
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制所有类别的ROC曲线
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'类别 {i} 的ROC曲线 (面积 = {roc_auc[i]:0.2f})')

# 对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('多类别接收者操作特性曲线')
plt.legend(loc="lower right")
plt.show()

# 保存模型
joblib.dump(best_model, 'random_forest_stress_level_model.pkl')