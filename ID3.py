import operator
import matplotlib
from math import log
import numpy as np
import plot

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataset, axis):
    retDataSet = []
    for featVec in dataset:
        reducedFeatVec = featVec[:axis]
        if (type(reducedFeatVec) != list):
            reducedFeatVec = reducedFeatVec.tolist()
        reducedFeatVec.extend(featVec[axis + 1:])
        retDataSet.append(reducedFeatVec)
    return retDataSet #去掉value对应特征

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1 #特征数
    baseEntropy = calcShannonEnt(dataset) #数据集的熵
    bestInfoGain = 0.0 #最佳信息增益
    bestFeature = -1 #最佳特征
    bestDivision = 0.0 #最佳特征的分割位置
    dataset = np.array(dataset)
    features = dataset[:, :-1]  # 特征
    for i in range(numFeatures):
        feature = sorted(features[:, i])
        divide = 0.0
        info_gain = 0.0
        for j in range(len(feature) - 1):
            new_divide = (feature[j] + feature[j + 1]) / 2
            left_subset = dataset[features[:, i] <= new_divide]
            right_subset = dataset[features[:, i] > new_divide]
            left_labels = left_subset[:, -1]
            right_labels = right_subset[:, -1]

            left_entropy = calcShannonEnt(left_subset)
            right_entropy = calcShannonEnt(right_subset)

            prob_left = len(left_subset) / len(dataset)
            prob_right = len(right_subset) / len(dataset)

            new_info_gain = baseEntropy - (prob_left * left_entropy + prob_right * right_entropy)
            if j == 0:
                info_gain = new_info_gain
                divide = new_divide
            elif new_info_gain > info_gain:
                info_gain = new_info_gain
                divide = new_divide
        if i == 0:
            bestInfoGain = info_gain
            bestFeature = i
            bestDivision = divide
        elif info_gain > bestInfoGain:
            bestInfoGain = info_gain
            bestFeature = i
            bestDivision = divide
    return bestFeature, bestDivision #返回最佳特征，以及最佳特征的分割位置

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] #返回出现次数最多的类别标签


def createTree(dataSet, labels, featLabels, depth = 0, max_depth = 5):
    classList = [example[-1] for example in dataSet]  # 取分类标签
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0 or depth >= max_depth:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat, bestDivison = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用特征标签
    dataSet = np.array(dataSet)
    myTree[bestFeatLabel]['<=' + str(bestDivison)] = createTree(splitDataSet(dataSet[dataSet[:, bestFeat] <= bestDivison], bestFeat), labels[:], featLabels, depth+1, max_depth)
    myTree[bestFeatLabel]['>' + str(bestDivison)] = createTree(splitDataSet(dataSet[dataSet[:, bestFeat] > bestDivison], bestFeat), labels[:], featLabels, depth+1, max_depth)
    return myTree

def classify(tree, feature_labels, test_instance):
    root = list(tree.keys())[0]  # 获取根节点
    child_nodes = tree[root]  # 获取子节点
    feature_index = feature_labels.index(root)  # 找到根节点对应的特征索引

    for key in child_nodes.keys():
        # 解析条件（例如 '<=2.5' 或 '>2.5'）
        condition = key.split('<=') if '<=' in key else key.split('>')
        threshold = float(condition[1])
        if ('<=' in key and test_instance[feature_index] <= threshold) or ('>' in key and test_instance[feature_index] > threshold):
            if isinstance(child_nodes[key], dict):
                return classify(child_nodes[key], feature_labels, test_instance)  # 递归分类
            else:
                return child_nodes[key]  # 返回叶节点（类别标签）

def calculate_accuracy(tree, feature_labels, test_data):
    correct_predictions = 0
    for instance in test_data:
        true_label = instance[-1]  # 最后一列为真实标签
        predicted_label = classify(tree, feature_labels, instance[:-1])  # 使用决策树预测
        if predicted_label == true_label:
            correct_predictions += 1
    accuracy = (correct_predictions / len(test_data)) * 100
    return accuracy

if __name__ == '__main__':
    dataset = []
    file_path = 'traindata.txt'
    with open(file_path, 'r') as file:
        for line in file:
            dataset.append([float(x) for x in line.strip().split('\t')])

    dataset = np.array(dataset)
    labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    labels1 = ['0', '1', '2']
    featLabels = []
    myTree = createTree(dataset, labels, featLabels)
    print(myTree)
    plot.create_plot(myTree)

# 加载测试数据
    test_data = []
    with open('testdata.txt', 'r') as file:
        for line in file:
            test_data.append([float(x) for x in line.strip().split('\t')])

    # 决策树对应的特征标签
    feature_labels = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    # 计算正确率
    accuracy = calculate_accuracy(myTree, feature_labels, test_data)
    print(f"正确率: {accuracy:.2f}%")