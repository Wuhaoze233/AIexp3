import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载示例数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 基尼指数决策树
gini_tree = DecisionTreeClassifier(
    criterion='gini',  # 基尼指数
    max_depth=3,
    random_state=42
)

# 训练模型
gini_tree.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(15,10))
plot_tree(
    gini_tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.show()

# 模型评估
print(f"基尼指数模型准确率: {gini_tree.score(X_test, y_test):.3f}")
print("特征重要性:")
for name, importance in zip(iris.feature_names, gini_tree.feature_importances_):
    print(f"{name}: {importance:.3f}")
