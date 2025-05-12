import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 解析数据（注意：原始数据需要保存为txt文件）
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            row = list(map(float, line.strip().split('\t')))
            data.append(row)
    return pd.DataFrame(data, columns=['sepal_length', 'sepal_width',
                                     'petal_length', 'petal_width', 'class'])


def feature_importance(model, feature_names):
    importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance['特征'], importance['重要性'])
    plt.title("特征重要性分析", fontsize=14)
    plt.xlabel("特征重要性权重")
    plt.gca().invert_yaxis()
    plt.show()



# 加载数据
train_df = load_data('traindata.txt')
test_df = load_data('testdata.txt')

# 分离特征和标签
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# 基尼指数决策树
gini_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2
)
gini_model.fit(X_train, y_train)

# 信息增益率决策树
entropy_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42,
    min_impurity_decrease=0.01
)
entropy_model.fit(X_train, y_train)

# 决策树可视化
plt.figure(figsize=(20, 12))
plot_tree(
    gini_model,
    feature_names=train_df.columns[:-1],
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    filled=True,
    rounded=True,
    proportion=True
)
plt.title("基尼指数决策树可视化", fontsize=16)
plt.show()
print("基尼指数模型特征重要性:")
feature_importance(gini_model, train_df.columns[:-1])

print("\n信息增益率模型特征重要性:")
feature_importance(entropy_model, train_df.columns[:-1])

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
#     print("\n分类报告:")
#     print(classification_report(y_test, y_pred))
#     print("\n混淆矩阵:")
#     print(confusion_matrix(y_test, y_pred))
#
#
# print("基尼指数模型评估:")
# evaluate_model(gini_model, X_test, y_test)
#
# print("\n信息增益率模型评估:")
# evaluate_model(entropy_model, X_test, y_test)