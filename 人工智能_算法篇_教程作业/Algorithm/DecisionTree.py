from math import log

class DecisionTree:
    # 决策树算法
    def createDataSet(self):  # 创建数据集
        # 数据集
        dataSet = [[0, 0, 0, 0, 'no'],
                   [0, 0, 0, 1, 'no'],
                   [0, 1, 0, 1, 'yes'],
                   [0, 1, 1, 0, 'yes'],
                   [0, 0, 0, 0, 'no'],
                   [1, 0, 0, 0, 'no'],
                   [1, 0, 0, 1, 'no'],
                   [1, 1, 1, 1, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [1, 0, 1, 2, 'yes'],
                   [2, 0, 1, 2, 'yes'],
                   [2, 0, 1, 1, 'yes'],
                   [2, 1, 0, 1, 'yes'],
                   [2, 1, 0, 2, 'yes'],
                   [2, 0, 0, 0, 'no']]
        labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 前四个数字（2，0，1，0）对应的标签
        return dataSet, labels  # 返回数据集和标签

    def calcShannonEnt(self, dataSet):  # 计算熵

        numEntires = len(dataSet)  # 数据集的行数（即用户个数）

        labels = [featVec[-1] for featVec in dataSet]  # 储存标签（dataSet最后一列）

        keys = set(labels)  # 标签种类（set去重）

        shannonEnt = 0.0  # 熵初始化
        for key in keys:
            prob = float(labels.count(key)) / numEntires  # 计算每种标签出现的几率
            shannonEnt -= prob * log(prob, 2)  # 计算熵
        return shannonEnt  # 返回熵

    def splitDataSet(self, dataSet, axis, value):  # 数据集分割(将第axis列中值为value的数据集去掉axis列)
        retDataSet = []  # 划分后的数据集
        for featVec in dataSet:  # 遍历原数据集
            if featVec[axis] == value:
                # 去掉选定的axis列
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])

                retDataSet.append(reducedFeatVec)  # 返回的数据集
        return retDataSet  # 返回划分后的数据集

    def chooseBestFeatureToSplit(self, dataSet):  # 找熵最小的维度（列）
        numFeatures = len(dataSet[0]) - 1  # 特征的数量(-1->去除label)
        baseEntropy = self.calcShannonEnt(dataSet)  # 原数据集的熵
        bestInfoGain = 0.0  # 信息增益初始化
        bestFeature = -1  # 最优特征的索引值 初始化
        for i in range(numFeatures):  # 遍历所有特征
            featList = [example[i] for example in dataSet]  # 获取dataSet的第i个特征所有行
            uniqueVals = set(featList)  # 去重
            newEntropy = 0.0  # 子集的熵(经验条件熵)
            for value in uniqueVals:  # 遍历特征中所有结果(例：年龄：1、2、3)
                subDataSet = self.splitDataSet(dataSet, i, value)  # 划分后的子集
                prob = len(subDataSet) / float(len(dataSet))  # 子集的概率
                newEntropy += prob * self.calcShannonEnt(subDataSet)  # 子集的熵
            infoGain = baseEntropy - newEntropy  # 信息增益
            # print("第%d个特征的增益为%.3f" % (i, infoGain))            #打印每个特征的信息增益
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
                bestFeature = i  # 记录信息增益最大的特征的索引值
        return bestFeature  # 返回信息增益最大的特征的索引值

    def majorityCnt(self, classList):  # 返回classList中出现次数最多的元素
        classCount = {}  # 创建空字典，用于计数
        keys = set(classLabel)  # 去重
        for key in keys:
            classCount[key] = classList.count(key)  # 记录各元素出现次数

        # 根据字典的值降序排序
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        return sortedClassCount[0][0]  # 返回出现次数最多的元素

    def createTree(self, dataSet, labels, lab_sel):  # 创建决策树
        classList = [example[-1] for example in dataSet]  # 取分类标签(yes/no)

        # 如果所有分类标签完全相同->完成划分
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        # 若遍历完所有特征(划分结束)->返回出现次数最多的类标签
        if len(dataSet[0]) == 1 or len(labels) == 0:
            return self.majorityCnt(classList)

        bestFeat = self.chooseBestFeatureToSplit(dataSet)  # 获取最优特征的维度
        bestFeatLabel = labels[bestFeat]  # 得到最优特征的标签
        lab_sel.append(labels[bestFeat])  # 最优特征的标签加入生成决策树需要的标签列表
        myTree = {bestFeatLabel: {}}  # 最优特征的标签作为根结点，子节点为空字典
        del (labels[bestFeat])  # 删除已经使用特征标签

        featValues = [example[bestFeat] for example in dataSet]  # 得到数据集中最优特征维度（某一列）的所有属性值
        uniqueVals = set(featValues)  # 去掉重复的属性值
        for value in uniqueVals:  # 遍历属性值
            subLabels = labels[:]  # 复制label[]中的所有元素
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels,
                                                           lab_sel)

        return myTree

    def classify(self, inputTree, featLabels, testVec):  # 进行分类
        '''决策树 inputTree、特征标签 featLabels 、测试向量 testVec'''
        firstStr = next(iter(inputTree))  # 获取决策树根结点
        secondDict = inputTree[firstStr]  # 根节点的子节点
        featIndex = featLabels.index(firstStr)  # 根节点对应的特征标签在 featLabels 中的索引位置
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    # 当前子节点的值为字典类型（即还有子节点），则递归调用 classify() 函数继续向下分类
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    # 当前子节点的值为叶节点（不再有子节点），则将叶节点的值作为分类结果
                    classLabel = secondDict[key]
        return classLabel  # 返回分类结果