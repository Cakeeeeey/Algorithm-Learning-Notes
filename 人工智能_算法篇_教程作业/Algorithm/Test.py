#其他模块
from KNN import KNN
from DecisionTree import DecisionTree
from CartTree import tree
from Bayes import Bayes
from LogisticRegression import LogisticRegression

#库
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#UI
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import uic

#中文防乱码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 切换为图形界面显示的终端TkAgg，防止报错has no attribute ‘FigureCanvas‘
matplotlib.use('TkAgg')

class Window:
    def __init__(self):
        self.app = QApplication([])  # 创建QApplication

        # 从文件中加载UI定义
        self.ui = uic.loadUi("UI/Algorithms.ui")
        self.ui.KNN.clicked.connect(self.knn_test)
        self.ui.DecisionTree.clicked.connect(self.DecisionTree_test)
        self.ui.CartTree.clicked.connect(self.CartTree_test)
        self.ui.Bayes.clicked.connect(self.Bayes_test)
        self.ui.LogisticRegression_test1.clicked.connect(self.LogisticRegression_test1)
        self.ui.LogisticRegression_test2.clicked.connect(self.LogisticRegression_test2)


    #KNNtest
    def knn_test(self):
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

    #决策树test
    def DecisionTree_test(self):
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

    #Cart分类树test
    def indexSplit(self,N, train_ratio):  # 将数据集分为训练数据train与测试数据test
        '''N-数据集总数 train_ratio-训练数据占数据集比例'''
        N_train = int(N * train_ratio)  # 训练数据数目
        index_random = np.random.permutation(N)  # 随机数
        index_train = index_random[:N_train]  # 随机选择训练数据索引值
        index_test = index_random[N_train:]  # 随机选择测试数据索引值

        return index_train, index_test  # 返回训练数据与测试数据的索引值

    def CartTree_test(self):

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

    #朴素贝叶斯test
    def Bayes_test(self):
        self.ui.Output.clear()  # 清空Output文本框

        '''实验一  自制贷款数据集'''
        self.ui.Output.append('自制贷款数据集：')

        # 创建Bayes的实例
        bayes=Bayes()

         # 获取数据集
        dataSet, labels = bayes.createDataSet()

        # 截取数据和标签
        datas = [i[:-1] for i in dataSet] #除最后一列，都是datas
        labs = [i[-1] for i in dataSet]  #最后一列是labs

        #获取标签种类
        keys = set(labs)

        # 进行模型训练
        model = bayes.trainPbmodel(datas,labs)
        self.ui.Output.append('朴素贝叶斯模型：')
        self.ui.Output.append(str(model))
        self.ui.Output.append('————————————————————————————')

        # 根据输入数据获得预测结果
        feat = [0,1,0,1]
        result = bayes.getPbfromModel(feat,model,keys)
        self.ui.Output.append('各结果概率：')
        self.ui.Output.append(str(result))
        self.ui.Output.append('————————————————————————————')

        # 遍历结果找到概率最大值进行数据
        for key,value in result.items():
            if(value == max(result.values())):
                self.ui.Output.append("预测结果："+key)
        self.ui.Output.append('————————————————————————————')

        '''实验二  隐形眼睛数据集'''
        self.ui.Output.append('隐形眼睛数据集：')
        # 读取数据文件 截取数据和标签
        with open("Statistics/Bayes/train_lenses.txt", 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()
        dataSet = [line.split('\t') for line in lines]

        datas = [i[:-1] for i in dataSet]
        labs = [i[-1] for i in dataSet]

        # 获取标签种类
        keys = set(labs)
        # 进行模型训练
        model = bayes.trainPbmodel(datas, labs)
        self.ui.Output.append('朴素贝叶斯模型：')
        self.ui.Output.append(str(model))
        self.ui.Output.append('————————————————————————————')

        # 测试
        # 读取测试文件
        with open("Statistics/Bayes/test_lenses.txt", 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()

        # 逐行读取数据并测试
        for line in lines:
            data = line.split('\t')
            lab_true = data[-1]
            feat = data[:-1]
            result = bayes.getPbfromModel(feat, model, keys)

            key_out = ""
            for key, value in result.items():
                if (value == max(result.values())):
                    key_out = key
            self.ui.Output.append("输入特征：")
            self.ui.Output.append(str(data))
            self.ui.Output.append("各结果概率：")
            self.ui.Output.append(str(result))
            self.ui.Output.append("预测结果： %s  医生推荐： %s" % (key_out, lab_true))

    #逻辑回归test
    def load_dataset(self,file):  # 读取文件
        # 分行读取
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()  # 文件每一行都作为line存储进lines

        # 读取标签
        labs = [line.split("\t")[-1] for line in lines]  # 每个line的最后一列[-1]存入labs
        labs = np.array(labs).astype(float)  # float类型
        labs = np.expand_dims(labs, axis=-1)  # labs维度拓展[N,]->[N,1],矢量变矩阵

        # 读取数据
        datas = [line.split("\t")[:-1] for line in lines]  # 每个line除最后一行[:-1]都存入datas
        datas = np.array(datas).astype(float)  # float类型
        N, D = np.shape(datas)  # 获取datas的维度

        datas = np.c_[np.ones([N, 1]), datas]  # datas增加一个全为1的维度(即z=XW中,与W0内积的维度)

        return datas, labs  # 返回读取并处理过的数据集、标签集

    def draw_desion_line(self,datas, labs, w): #绘制数据点和判决线

        #设定0、1各自对应的颜色
        #(0.8, 0, 0)->(80% 红色，0% 绿色，0% 蓝色)
        dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0)}

        # 画数据点
        for i in range(2): #绘制0、1数据点
            index = np.where(labs == i)[0] #取出值为i的数据点的索引值

            sub_datas = datas[index] #取出值为i的数据点

            #第2列为x轴，第3列为y轴，画出数据点(datas第1列都是'1')
            plt.scatter(sub_datas[:, 1], sub_datas[:, 2], s=16., color=dic_colors[i])

        # 画判决线
        min_x = np.min(datas[:, 1]) #第2列为x轴，取最小值
        max_x = np.max(datas[:, 1]) #第2列为x轴，取最大值
        w = w[:, 0] #取权重第一列
        x = np.arange(min_x, max_x, 0.01) #从min到max生成数组，步长为0.01
        y = -(x * w[1] + w[0]) / w[2] #判决线公式(由h=0.5推导而来)
        plt.plot(x, y) #画判决线

        plt.show()

    def LogisticRegression_test1(self):
        ''' 实验1 基础测试数据'''
        self.ui.Output.clear()  # 清空Output文本框
        logisticRegression=LogisticRegression() #LogisticRegression实例化

        # 加载数据
        file = "Statistics/LogisticRegression/testset.txt"
        datas, labs = self.load_dataset(file)

        weights,courses= logisticRegression.train_LR(datas, labs, alpha=0.002, n_epoch=400) #训练模型，得到合适的权重
        #alpha=0.0001 n_epoch=8000,error∈[2,3]%
        self.ui.Output.append(''' ——————实验1 基础测试数据——————'''+'\n'+'\n'+'迭代过程：')
        self.ui.Output.append(str(courses))
        self.ui.Output.append('\n'+'最终权重为：')
        self.ui.Output.append(str(weights.tolist()))

        #绘制数据点与判决线
        labs = np.array(labs)  # 将 labs 转换为 NumPy 数组
        self.draw_desion_line(datas, labs, weights)

    def LogisticRegression_test2(self):
        '''实验2 马疝数据集'''
        self.ui.Output.clear()  # 清空Output文本框
        logisticRegression=LogisticRegression() #LogisticRegression实例化
        # 加载训练数据
        train_file = "horse_train.txt"
        train_datas, train_labs = self.load_dataset(train_file)

        # 加载测试数据
        test_file = "horse_test.txt"
        test_datas, test_labs = self.load_dataset(test_file)

        # # 梯度下降
        # weights = train_LR(train_datas,train_labs,alpha=0.001,n_epoch=90)
        # print(weights)

        # 随机梯度下降
        weights,courses = logisticRegression.train_LR_batch(train_datas, train_labs, batchsize=2, n_epoch=34, alpha=0.001)

        self.ui.Output.append(''' ——————实验2 马疝数据集——————''' + '\n' + '\n' + '迭代过程：')
        self.ui.Output.append(str(courses))
        self.ui.Output.append('\n' + '最终权重为：')
        self.ui.Output.append(str(weights.tolist()))

        # 截取几个维度画图
        index = [0, 4, 5]
        sub_datas = train_datas[:, index]
        sub_weights = weights[index]
        self.draw_desion_line(sub_datas, train_labs, sub_weights)
        plt.show()
        





app = QApplication([])
window = Window()  # Window类的实例化
window.ui.show()
app.exec()
