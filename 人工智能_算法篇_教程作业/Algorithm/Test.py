from KNN import KNN
from DecisionTree import DecisionTree
from CartTree import tree
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
        self.ui.CartTree.clicked.connect(self.CartTree_practice)

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

        Knn = KNN()  # 创建 KNN 类的实例
        k = 5
        n_right = 0  # 记录预测正确次数
        for i in range(N_test):
            test = data_test[i, :]
            det = Knn.knn(data_train, test, lab_train, k)  # 预测结果

            if det == lab_test[i]:
                n_right += 1  # 预测正确
                # 将输出追加到Output文本框
                output = '样本 %d 实际种类=%s 预测种类：%s' % (i, lab_test[i], det)
                self.ui.Output.append(output)
        # 分析预测准确度
        accuracy = n_right / N_test * 100
        self.ui.Output.append('准确度为%.2f %%' % accuracy)

    #决策树实践
    def DecisionTree_practice(self):
        self.ui.Output.clear()  # 清空Output文本框

        decisiontree=DecisionTree() #实例化

        ''' 实验一 自定义贷款数据集  '''
        dataSet, labels = decisiontree.createDataSet()  # 获取数据集
        lab_copy = labels[:]
        lab_sel = []
        myTree = decisiontree.createTree(dataSet, labels, lab_sel)
        self.ui.Output.append('实验一 自定义贷款数据集'+'\n'+'决策树：'+str(myTree))

        # 测试
        testVec = [0, 1, 1, 2]
        result = decisiontree.classify(myTree, lab_copy, testVec)
        self.ui.Output.append('分类结果：'+result+'\n'+'————————————————')

        ''' 实验二  隐形眼睛数据集 '''
        with open("Statistics/DecisionTree/train-lenses.txt",'r',encoding='utf-8') as f:
            lines = f.read().splitlines()

        dataSet = [line.split('\t') for line in lines]
        labels = ['年龄','近视/远视','是否散光','是否眼干']

        lab_copy = labels[:]
        lab_sel = []
        myTree = decisiontree.createTree(dataSet, labels,lab_sel)
        self.ui.Output.append('实验二  隐形眼睛数据集'+'\n'+'决策树：' + str(myTree))

        # 测试
        with open("Statistics/DecisionTree/test-lenses.txt",'r',encoding='utf-8') as f:
            lines = f.read().splitlines()

        for line in lines:
            data = line.split('\t')
            lab_true = data[-1]
            test_vec = data[:-1]
            result = decisiontree.classify(myTree,lab_copy,test_vec)
            self.ui.Output.append("预测结果: %s  医生推荐: %s" % (result, lab_true))

    #Cart分类树实践
    def indexSplit(self,N, train_ratio):  # 将数据集分为训练数据train与测试数据test
        '''N-数据集总数 train_ratio-训练数据占数据集比例'''
        N_train = int(N * train_ratio)  # 训练数据数目
        index_random = np.random.permutation(N)  # 随机数
        index_train = index_random[:N_train]  # 随机选择训练数据索引值
        index_test = index_random[N_train:]  # 随机选择测试数据索引值

        return index_train, index_test  # 返回训练数据与测试数据的索引值

    def CartTree_practice(self):

        self.ui.Output.clear()  # 清空Output文本框

        # 导入数据集
        file_data = 'Statistics\CartTree\iris.data'

        # 数据读取
        datas = np.loadtxt(file_data, dtype=float, delimiter=',', usecols=(0, 1, 2, 3))
        labs = np.loadtxt(file_data, dtype=str, delimiter=',', usecols=(4))
        N, D = np.shape(datas)

        # 分为训练集和测试集
        index_train, index_test = self.indexSplit(N, train_ratio=0.6)

        train_datas = datas[index_train, :]
        train_labs = labs[index_train]

        test_datas = datas[index_test, :]
        test_labs = labs[index_test]

        stopping_sz = 1  # 分类到数据量为1时停止

        decision_tree_classifier = tree(train_datas, train_labs, stopping_sz)  # 类型Tree
        decision_tree_classifier.fit()  # 创建分类树
        ret_tree = decision_tree_classifier.print_tree()  # 输出分类树
        self.ui.Output.append('分类树:'+str(ret_tree)+'\n' + '——————————————————————————————')

        # 计算预测准确度
        n_right = 0  # 记录预测正确的数量
        for i in range(test_datas.shape[0]):
            prediction = decision_tree_classifier.predict(test_datas[i])

            if prediction == test_labs[i]:
                n_right = n_right + 1

            self.ui.Output.append('预测标签=' + prediction+'实际标签=' + test_labs[i])

        self.ui.Output.append('———————————————————————————')
        self.ui.Output.append("准确率=%.2f%%" % (n_right * 100 / len(test_labs)))


app = QApplication([])
window = Window()  # Window类的实例化
window.ui.show()
app.exec()
