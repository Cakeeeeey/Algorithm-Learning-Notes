import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def sigmoid(self,z):  # 预测函数
        return 1.0 / (1 + np.exp(-z))

    def weight_update(self,datas, labs, w, alpha=0.01): #计算权重w
        # datas 数据
        # labs 标签
        # w    权重
        z = np.dot(datas, w)  #datas与w的矩阵内积
        h = self.sigmoid(z)  # 概率=预测函数
        Error = labs - h  # N误差=实际标签-预测标签
        w = w + alpha * np.dot(datas.T, Error) #计算权重w
        return w

    def test_accuracy(self,datas, labs, w): #计算某权重下的失误率
        N, D = np.shape(datas) #获取datas的维度
        z = np.dot(datas, w)  #datas与w的矩阵内积
        h = self.sigmoid(z)  #概率=预测函数
        lab_det = (h > 0.5).astype(float) #预测标签(>0.5视为1)
        error_rate = np.sum(np.abs(labs - lab_det)) / N #失误率(|预测与实际的差|之和)
        return error_rate

    def train_LR(self,datas, labs, n_epoch=2, alpha=0.005): #训练逻辑回归模型(权重w的迭代)
        courses=[] #记录迭代过程
        #alpha 迭代步长
        #n_epoch 迭代次数
        N, D = np.shape(datas) #获取datas维度
        w = np.ones([D, 1])  # w初始化为1，且维度与datas一致

        # 进行n_epoch轮迭代
        for i in range(n_epoch):
            w = self.weight_update(datas, labs, w, alpha) #计算权重w
            error_rate = self.test_accuracy(datas, labs, w) #计算失误率
            courses.append("迭代次数 %d 失误率 %.3f%%" % (i, error_rate * 100))
        return w,courses #返回迭代后的w与迭代过程courses

    def train_LR_batch(self,datas, labs, batchsize, n_epoch=2, alpha=0.005): # 随机梯度下降
        courses = []  # 记录迭代过程
        N, D = np.shape(datas) #获取数据维度

        # weight 初始化
        w = np.ones([D, 1])  # Dx1

        #计算batch数量
        N_batch = N // batchsize #数据数量N对batchsize取整

        #迭代
        for i in range(n_epoch):
            # 打乱数据
            rand_index = np.random.permutation(N).tolist()

            # 每个batch 更新一下weight
            for j in range(N_batch):
                # 遍历batch
                index = rand_index[j * batchsize:(j + 1) * batchsize]

                batch_datas = datas[index]
                batch_labs = labs[index]

                #计算权重
                w = self.weight_update(batch_datas, batch_labs, w, alpha)

            error = self.test_accuracy(datas, labs, w)
            courses.append("迭代次数 %d 失误率 %.3f%%" % (i, error * 100))
        return w,courses