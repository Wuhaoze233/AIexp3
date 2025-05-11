from math import log

def preprocessDataSet(): #将数据集中原本的离散型变量更换为离散型变量
    pass

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

def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet #去掉value对应特征

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1 #特征数
    baseEntropy = calcShannonEnt(dataset) #数据集的熵
    bestInfoGain = 0.0 #最佳信息增益
    bestFeature = -1 #最佳特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataset] #第i个特征的所有值
        uniqueVals = set(featList) #获取单个特征
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset,i,value) #划分数据集
            prob = len(subDataSet) / float(len(dataset)) #子集中概率
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature #返回最佳特征

