import numpy as np

class Bayes:
    def createDataSet(self):
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
        labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
        return dataSet, labels  # 返回数据集和分类属性

    def trainPbmodel_X(self,feats):  # 获取概率模型(计算feat中各值出现的概率)
        N, D = np.shape(feats)  # 数据集大小
        model = {}  # 创建空模型
        # 对每一维度的特征进行概率统计
        for d in range(D):  # 遍历每一列(即每一个特征)
            data = feats[:, d].tolist()  # 取第d列，转换为list格式
            keys = set(data)  # 数据去重
            N = len(data)  # 行数
            model[d] = {}  # 创建模型中键值为d的空间
            for key in keys:  # 遍历去重后的数据
                model[d][key] = float(data.count(key) / N)  # 计算特征中各值出现的概率
        return model  # 返回概率模型

    def trainPbmodel(self,datas, labs):
        # datas： list格式 每个元素表示1个特征序列
        # labs：  list格式 每个元素表示一个标签
        model = {}  # 创建空模型
        keys = set(labs)  # 获取标签的类别
        for key in keys:  # 遍历个标签
            Pbmodel_Y = labs.count(key) / len(labs)  # 获得P(Y)
            index = np.where(np.array(labs) == key)[0].tolist()  # 标签为Y的数据的索引值
            feats = np.array(datas)[index]  # 标签为Y的数据

            Pbmodel_X = self.trainPbmodel_X(feats)  # 获得 P(X|Y)

            # 模型保存
            model[key] = {}
            model[key]["PY"] = Pbmodel_Y
            model[key]["PX"] = Pbmodel_X
        return model

    def getPbfromModel(self,feat, model, keys):  # 计算result(即似然函数)
        # feat : list格式 一条输入特征
        # model: 训练的概率模型
        # keys ：考察标签的种类
        results = {}
        eps = 0.00001  # 一个很小的数，在log中代替0防止数学错误
        for key in keys:  # 遍历特征的各标签
            PY = model.get(key, eps).get("PY")  # 从模型中获取P(Y)，若P(Y)不存在则返回eps(即0)

            model_X = model.get(key, eps).get("PX")  # 从模型中获取 P(X|Y)，若P(X|Y)不存在则返回eps(即0)

            list_px = []  # 根据输入的feat计算P(X|Y)
            for d in range(len(feat)):
                pb = model_X.get(d, eps).get(feat[d], eps)
                list_px.append(pb)

            result = np.log(PY) + np.sum(np.log(list_px))  # 似然函数
            results[key] = result
        return results