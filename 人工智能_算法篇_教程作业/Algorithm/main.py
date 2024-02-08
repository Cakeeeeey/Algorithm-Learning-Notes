import operator
from math import log
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import uic

class Window:
    def __init__(self):
        self.app = QApplication([])  # 创建QApplication

        # 从文件中加载UI定义
        self.ui = uic.loadUi("UI/Algorithms.ui")
        self.ui.KNN.clicked.connect(self.knn_practice)
        self.ui.DecisionTree.clicked.connect(self.DecisionTree_practice)

    #KNN算法
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

    #KNN实践
    def knn_practice(self):
        # knn示例
        self.ui.Output.clear()  # 清空Output文本框

        file_data = 'Statistics/KNN/iris.data'

        data = np.loadtxt(file_data, dtype=np.float64, delimiter=',', usecols=(0, 1, 2, 3))  # 读取前三列
        lab = np.loadtxt(file_data, dtype=str, delimiter=',', usecols=(4))  # 读取第四行

        N = 150  # 总数据
        N_train = 100  # 训练数据
        N_test = N - N_train  # 测试数据
        perm = np.random.permutation(N)  # 打乱总数据

        index_train = perm[:N_train]  # 前100个作为训练数据
        index_test = perm[N_train:]  # 除了前100个，其他的作为测试数据

        data_train = data[index_train, :]  # 训练集数据
        lab_train = lab[index_train]  # 训练集标签
        data_test = data[index_test, :]  # 测试集数据
        lab_test = lab[index_test]  # 测试集标签

        k = 5
        n_right = 0  # 记录预测正确次数
        for i in range(N_test):
            test = data_test[i, :]
            det = self.knn(data_train, test, lab_train, k)  # 预测结果

            if det == lab_test[i]:
                n_right += 1  # 预测正确
                # 将输出追加到Output文本框
                output = '样本 %d 实际种类=%s 预测种类：%s' % (i, lab_test[i], det)
                self.ui.Output.append(output)
        # 分析预测准确度
        accuracy = n_right / N_test * 100
        self.ui.Output.append('准确度为%.2f %%' % accuracy)

    #决策树算法
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
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels, lab_sel)

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

    #决策树实践
    def DecisionTree_practice(self):
        self.ui.Output.clear()  # 清空Output文本框

        ''' 实验一 自定义贷款数据集  '''
        dataSet, labels = self.createDataSet()  # 获取数据集
        lab_copy = labels[:]
        lab_sel = []
        myTree = self.createTree(dataSet, labels, lab_sel)
        self.ui.Output.append('实验一 自定义贷款数据集'+'\n'+'决策树：'+str(myTree))

        # 测试
        testVec = [0, 1, 1, 2]
        result = self.classify(myTree, lab_copy, testVec)
        self.ui.Output.append('分类结果：'+result+'\n'+'————————————————')

        ''' 实验二  隐形眼睛数据集 '''
        with open("Statistics/DecisionTree/train-lenses.txt",'r',encoding='utf-8') as f:
            lines = f.read().splitlines()

        dataSet = [line.split('\t') for line in lines]
        labels = ['年龄','近视/远视','是否散光','是否眼干']

        lab_copy = labels[:]
        lab_sel = []
        myTree = self.createTree(dataSet, labels,lab_sel)
        self.ui.Output.append('实验二  隐形眼睛数据集'+'\n'+'决策树：' + str(myTree))

        # 测试
        with open("Statistics/DecisionTree/test-lenses.txt",'r',encoding='utf-8') as f:
            lines = f.read().splitlines()

        for line in lines:
            data = line.split('\t')
            lab_true = data[-1]
            test_vec = data[:-1]
            result = self.classify(myTree,lab_copy,test_vec)
            self.ui.Output.append("预测结果: %s  医生推荐: %s" % (result, lab_true))

    #Cart分类树算法
    def get_possible_splits(self,datas, ind_fea):  # 从datas 的第 ind_fea 维特征中获取所有可能得分割阈值
        feas = datas[:, ind_fea] #取第 ind_fea 维特征的所有属性
        feas = np.unique(feas) #属性去重
        feas = np.sort(feas) #属性升序排序
        splits = [] #创建分割阈值集
        #计算分割阈值
        for i in range(len(feas) - 1):
            th = (feas[i] + feas[i + 1]) / 2
            splits.append(th)
        return np.array(splits) #返回分割阈值集
    def gini_impurity(self,labs): #计算基尼系数
        unique_labs = np.unique(labs) #标签去重
        gini = 0 #基尼系数初始化
        for lab in unique_labs: #遍历各类标签
            n_pos = np.where(labs == lab)[0].shape[0] #各标签出现次数
            prob_pos = n_pos / len(labs) #各标签出现概率
            gini += prob_pos ** 2 #计算基尼系数-1
        gini = 1 - gini #计算基尼系数-2
        return gini #返回基尼系数
    def eval_split(self,datas, labs, ind_fea, split):  # 计算 datas 的 ind_fea 维的基尼增益
        mask = datas[:, ind_fea] <= split  #是否小于分割阈值
        index_l = np.where(mask == 1)[0]  #左侧特征索引值(小于分割阈值)
        index_r = np.where(mask == 0)[0]  #右侧特征索引值(大于分割阈值)
        labs_l = labs[index_l]  #左侧特征标签
        labs_r = labs[index_r]  #右侧特征标签

        weight_left = float(len(labs_l) / len(labs)) #左侧权重
        weight_right = 1 - weight_left  #右侧权重

        gini_parent =self.gini_impurity(labs)  #总基尼系数
        gini_left = self.gini_impurity(labs_l)  #左侧基尼系数
        gini_right = self.gini_impurity(labs_r)  #右侧基尼系数
        weighted_gini = gini_parent - (weight_left * gini_left + weight_right * gini_right) #计算基尼增益

        return weighted_gini #返回基尼增益
    class node: #节点类
        def __init__(self, datas, labs, parent):
            #处理输入的变量
            self.parent = parent
            self.datas = datas
            self.labs = labs

            # 当前节点的基尼系数
            self.gini = self.gini_impurity(self.labs)

            # 左右子树初始化
            self.left = None
            self.right = None

            # 当前节点的分割条件
            self.splitting_ind_fea = None #待分割的特征
            self.threshold = 0 #分割阈值

            self.leaf = False #叶节点判断 初始化
            self.label = None #当前节点标签 初始化
            self.confidence = None #叶节点标签的纯度(即该结果的置信度)
        def set_splitting_criteria(self, ind_fea, threshold):# 设置当前节点的分割条件
            self.splitting_ind_fea = ind_fea
            self.threshold = threshold
        def is_leaf(self, stopping_sz): #是否为叶节点(即是否停止分割)
            #剩下的数据小于stopping_sz/基尼系数为0(即所有数据标签相同)->停止分割
            if len(self.labs) <= stopping_sz or self.gini == 0.0:
                return True
            else:
                return False
        def find_splitting_criterion(self):# 找到当前节点 最佳的分割维度(ind_fea) 以该维度最佳的分割阈值
            max_score = -1.0 #最佳基尼系数初始化
            best_ind_fea = None #最佳维度初始化
            threshold = 0.0 #最佳分割阈值初始化

            dim_fea = np.shape(self.datas)[-1] #数据集的特征数(即列数)

            for i in range(dim_fea): #遍历各特征
                splits = self.get_possible_splits(self.datas, i) #分割阈值 集

                for split in splits: #遍历各分割阈值
                    split_score = self.eval_split(self.datas, self.labs, i, split) #计算基尼增益
                    if split_score > max_score: #记录最大的基尼增益
                        max_score = split_score
                        best_ind_fea = i
                        threshold = split

            return max_score, best_ind_fea, threshold #返回最大基尼增益、最佳分割维度、最佳分割阈值
        def split(self, ind_fea, threshold): # 对当前的节点进行分割

            mask = self.datas[:, ind_fea] <= threshold #是否小于等于分割阈值

            #处理输入的变量
            index_l = np.where(mask == 1)[0]  #左侧特征的索引值
            index_r = np.where(mask == 0)[0]  #右侧特征的索引值
            labs_l = self.labs[index_l]#左侧特征的标签
            labs_r = self.labs[index_r]#右侧特征的标签
            datas_l = self.datas[index_l, :]#左侧特征
            datas_r = self.datas[index_r, :]#右侧特征

            #输出分割结果
            print("将 %d 分割为 %d 与 %d 特征索引值： %d 分割阈值：%.2f" % (
            len(self.labs), len(labs_l), len(labs_r), ind_fea, threshold))

            #左右子树
            left = node(datas_l, labs_l, self)
            right = node(datas_r, labs_r, self)

            return left, right #返回左右子树
        def set_as_leaf(self):# 将当前节点设为叶子节点

            self.leaf = True #设置该节点为叶节点


            labs = self.labs.tolist() #转换为list变量，便于计算
            self.label = max(labs, key=labs.count) #设置该节点的标签为数据中数量最多的标签

            n_pos = len(np.where(self.labs == self.label)[0]) #标签与该节点相同的数量
            self.confidence = float(n_pos / len(self.labs)) #计算标签纯度
    class tree:

        def __init__(self, datas, labs, stopping_sz):

            self.root = None #根结点初始化

            #处理输入的变量
            self.datas = datas
            self.labs = labs
            self.stopping_sz = stopping_sz

            self.dic_tree = {} #创建空决策树

        def __build_tree(self, root):

            # 如果是叶子节点则返回
            if root.is_leaf(self.stopping_sz):
                root.set_as_leaf()
                return

            # 如果不是叶子节点，则找最佳分割(基尼增益、最佳特征索引值、最佳分割阈值)
            max_score, best_ind_fea, threshold = root.find_splitting_criterion()

            #没找到最佳特征索引值则返回
            if best_ind_fea == None:
                return

            # 设置分割条件
            root.set_splitting_criteria(best_ind_fea, threshold)

            # 对当前节点进行分割
            left, right = root.split(best_ind_fea, threshold)
            root.left = left
            root.right = right

            #递归分割左右子树
            self.__build_tree(root.left)
            self.__build_tree(root.right)
            return


app = QApplication([])
window = Window()  # Window类的实例化
window.ui.show()
app.exec()
