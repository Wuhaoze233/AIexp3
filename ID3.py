# import numpy as np
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# # 解析数据（注意：原始数据需要保存为txt文件）
# def load_data(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines[1:]:  # 跳过表头
#             row = list(map(float, line.strip().split('\t')))
#             data.append(row)
#     return pd.DataFrame(data, columns=['sepal_length', 'sepal_width',
#                                      'petal_length', 'petal_width', 'class'])
#
#
# def feature_importance(model, feature_names):
#     importance = pd.DataFrame({
#         '特征': feature_names,
#         '重要性': model.feature_importances_
#     }).sort_values('重要性', ascending=False)
#
#     plt.figure(figsize=(10, 6))
#     plt.barh(importance['特征'], importance['重要性'])
#     plt.title("特征重要性分析", fontsize=14)
#     plt.xlabel("特征重要性权重")
#     plt.gca().invert_yaxis()
#     plt.show()
#
#
#
# # 加载数据
# train_df = load_data('traindata.txt')
# test_df = load_data('testdata.txt')
#
# # 分离特征和标签
# X_train = train_df.iloc[:, :-1].values
# y_train = train_df.iloc[:, -1].values
# X_test = test_df.iloc[:, :-1].values
# y_test = test_df.iloc[:, -1].values
#
# # 基尼指数决策树
# gini_model = DecisionTreeClassifier(
#     criterion='gini',
#     max_depth=3,
#     random_state=42,
#     min_samples_split=5,
#     min_samples_leaf=2
# )
# gini_model.fit(X_train, y_train)
#
# # 信息增益率决策树
# entropy_model = DecisionTreeClassifier(
#     criterion='entropy',
#     max_depth=4,
#     random_state=42,
#     min_impurity_decrease=0.01
# )
# entropy_model.fit(X_train, y_train)
#
# # 决策树可视化
# plt.figure(figsize=(20, 12))
# plot_tree(
#     gini_model,
#     feature_names=train_df.columns[:-1],
#     class_names=['Setosa', 'Versicolor', 'Virginica'],
#     filled=True,
#     rounded=True,
#     proportion=True
# )
# plt.title("基尼指数决策树可视化", fontsize=16)
# plt.show()
# print("基尼指数模型特征重要性:")
# feature_importance(gini_model, train_df.columns[:-1])
#
# print("\n信息增益率模型特征重要性:")
# feature_importance(entropy_model, train_df.columns[:-1])
#
# # def evaluate_model(model, X_test, y_test):
# #     y_pred = model.predict(X_test)
# #     print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
# #     print("\n分类报告:")
# #     print(classification_report(y_test, y_pred))
# #     print("\n混淆矩阵:")
# #     print(confusion_matrix(y_test, y_pred))
# #
# #
# # print("基尼指数模型评估:")
# # evaluate_model(gini_model, X_test, y_test)
# #
# # print("\n信息增益率模型评估:")
# # evaluate_model(entropy_model, X_test, y_test)

# 导入必要库
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# 假设数据文件为CSV格式（根据实际情况调整读取方式）
data = pd.read_csv('traindata.txt', sep='\t', names=['a', 'b', 'c', 'd', 'class'], header=None)  # 或使用其他读取方式
print(data.columns[:-1])
# 查看数据基本信息
print("数据基本信息:")
print(f"样本量: {data.shape[0]}")
print(f"特征数: {data.shape[1]-1}")
print(f"类别分布:\n{data.iloc[:, -1].value_counts()}")

# 2. 模型构建
# 创建基尼指数决策树分类器（关键参数）
clf = DecisionTreeClassifier(
    criterion='gini',        # 使用基尼系数
    max_depth=10,            # 最大树深度（可调整）
    random_state=42,        # 结果可复现
    min_samples_split=2,    # 节点分裂最小样本数
    min_samples_leaf=1      # 叶子节点最小样本数
)

# 直接使用全部数据训练模型
clf.fit(data.iloc[:, :-1], data.iloc[:, -1])

# 3. 决策树可视化
plt.figure(figsize=(20, 10))  # 设置绘图尺寸

# 使用sklearn内置绘图函数
plot_tree(
    clf,
    feature_names=data.columns[:-1],
    class_names=data.iloc[:, -1].unique().astype(str),
    filled=True,            # 填充颜色
    rounded=True,           # 圆角框
    proportion=True,        # 显示比例
    impurity=True          # 显示不纯度
)

plt.title("Decision Tree Visualization (Gini Index)", fontsize=20)
plt.show()

from sklearn.metrics import accuracy_score, classification_report

# 加载测试数据
test_data = pd.read_csv('testdata.txt', sep='\t', names=['a', 'b', 'c', 'd', 'class'], header=None)

# 分离特征和标签
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# 使用模型进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 输出分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=data.iloc[:, -1].unique().astype(str)))
# 4. 可选：生成Graphviz格式文件（需安装graphviz）
# from sklearn.tree import export_graphviz
# export_graphviz(
#     clf,
#     out_file='tree.dot',
#     feature_names=data.columns[:-1],
#     class_names=data.columns[-1].unique().astype(str),
#     filled=True,
#     rounded=True
# )
# # 生成PNG图片（需graphviz环境）
# # !dot -Tpng tree.dot -o tree.png
