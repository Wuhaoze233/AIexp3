import operator
import matplotlib
from math import log
import numpy as np

def preprocessDataSet(file_path): #将数据集中原本的离散型变量更换为离散型变量
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            dataset.append([float(x) for x in line.strip().split('\t')])

    dataset = np.array(dataset)
    features = dataset[:, :-1]  # 特征
    labels = dataset[:, -1]  # 标签

    processed_feature = []
    division = []
    for i in range(features.shape[1]):
        feature = np.sort(features[:, i])
        divide = 0.0
        info_gain = 0.0
        for j in range(len(feature) - 1):
            new_divide = (feature[j] + feature[j + 1]) / 2
            left_subset = features[features[:, i] <= new_divide]
            right_subset = features[features[:, i] > new_divide]
            left_labels = labels[features[:, i] <= new_divide]
            right_labels = labels[features[:, i] > new_divide]

            left_entropy = calcShannonEnt(np.column_stack((left_subset, left_labels)))
            right_entropy = calcShannonEnt(np.column_stack((right_subset, right_labels)))

            prob_left = len(left_subset) / len(features)
            prob_right = len(right_subset) / len(features)

            new_info_gain = calcShannonEnt(np.column_stack((features, labels))) - (
                    prob_left * left_entropy + prob_right * right_entropy
            )
            if j == 0:
                info_gain = new_info_gain
                divide = new_divide
            elif new_info_gain > info_gain:
                info_gain = new_info_gain
                divide = new_divide
        division.append(divide)
        processed_feature.append(np.where(features[:, i] <= divide, 0, 1))
    processed_feature = np.array(processed_feature).T
    processed_dataset = np.column_stack((processed_feature, labels))
    return processed_dataset, division


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
    # for i in range(numFeatures):
    #     featList = [example[i] for example in dataset] #第i个特征的所有值
    #     uniqueVals = set(featList) #获取单个特征
    #     newEntropy = 0.0
    #     for value in uniqueVals:
    #         subDataSet = splitDataSet(dataset,i,value) #划分数据集
    #         prob = len(subDataSet) / float(len(dataset)) #子集中概率
    #         newEntropy += prob * calcShannonEnt(subDataSet)
    #     infoGain = baseEntropy - newEntropy
    #     if (infoGain > bestInfoGain):
    #         bestInfoGain = infoGain
    #         bestFeature = i
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
    # featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    # uniqueVals = set(featValues)  # 去掉重复的属性值
    # for value in uniqueVals:  # 遍历特征，创建决策树。
    #     subLabels = labels[:]
    #     myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
    dataSet = np.array(dataSet)
    myTree[bestFeatLabel]['<=' + str(bestDivison)] = createTree(splitDataSet(dataSet[dataSet[:, bestFeat] <= bestDivison], bestFeat), labels[:], featLabels, depth+1, max_depth)
    myTree[bestFeatLabel]['>' + str(bestDivison)] = createTree(splitDataSet(dataSet[dataSet[:, bestFeat] > bestDivison], bestFeat), labels[:], featLabels, depth+1, max_depth)
    return myTree


if __name__ == '__main__':
    # dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
    #            [0, 0, 0, 1, 'no'],
    #            [0, 1, 0, 1, 'yes'],
    #            [0, 1, 1, 0, 'yes'],
    #            [0, 0, 0, 0, 'no'],
    #            [1, 0, 0, 0, 'no'],
    #            [1, 0, 0, 1, 'no'],
    #            [1, 1, 1, 1, 'yes'],
    #            [1, 0, 1, 2, 'yes'],
    #            [1, 0, 1, 2, 'yes'],
    #            [2, 0, 1, 2, 'yes'],
    #            [2, 0, 1, 1, 'yes'],
    #            [2, 1, 0, 1, 'yes'],
    #            [2, 1, 0, 2, 'yes'],
    #            [2, 0, 0, 0, 'no']]
    #
    # labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    # labels1 = ['放贷', '不放贷']
    # featLabels = []
    # myTree = createTree(dataSet, labels, featLabels)
    # print(myTree)
    # dataset = []
    # division = []
    # dataset, division = preprocessDataSet('traindata.txt')
    # print(dataset)
    # print(division)
    dataset = []
    file_path = 'traindata.txt'
    with open(file_path, 'r') as file:
        for line in file:
            dataset.append([float(x) for x in line.strip().split('\t')])

    dataset = np.array(dataset)

    print(dataset)
    labels = ['a1', 'b2', 'c3', 'd4']
    labels1 = ['0', '1', '2']
    featLabels = []
    myTree = createTree(dataset, labels, featLabels)
    print(myTree)
