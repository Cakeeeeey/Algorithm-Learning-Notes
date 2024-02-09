import operator
import numpy as np
# KNN算法
class KNN:
    def knn(self, trainData, testData, labels, k):
        '''
        trainData-训练集 N,D N行D列
        testData-测试1,D 1行D列
        labels-训练集标签
        '''
        rowSize = trainData.shape[0]  # 训练样本行数
        diff = np.tile(testData, (rowSize, 1)) - trainData  # 训练样本（扩展后）与测试样本的差值
        sqrDiff = diff ** 2  # 训练样本（扩展后）与测试样本的平方差
        sqrDiffSum = sqrDiff.sum(axis=1)  # 训练样本（扩展后）与测试样本的平方差之和
        distances = sqrDiffSum ** 0.5  # 训练样本（扩展后）与测试样本的距离（平方差之和再开根号）
        sortDistance = distances.argsort()  # 对距离从小到大排序

        count = {}  # 存储labels出现的次数
        for i in range(k):  # 遍历距离最近的k个训练样本
            vote = labels[sortDistance[i]]  # 当前测试样本的label
            count[vote] = count.get(vote, 0) + 1  # labels计数

        sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)  # 对count降序排列
        return sortCount[0][0]  # 返回出现次数最多的label
