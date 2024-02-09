import numpy as np

# Cart分类树算法
def get_possible_splits(datas, ind_fea):  # 从datas 的第 ind_fea 维特征中获取所有可能得分割阈值
    feas = datas[:, ind_fea]  # 取第 ind_fea 维特征的所有属性
    feas = np.unique(feas)  # 属性去重
    feas = np.sort(feas)  # 属性升序排序
    splits = []  # 创建分割阈值集
    # 计算分割阈值
    for i in range(len(feas) - 1):
        th = (feas[i] + feas[i + 1]) / 2
        splits.append(th)
    return np.array(splits)  # 返回分割阈值集


def gini_impurity(labs):  # 计算基尼系数
    unique_labs = np.unique(labs)  # 标签去重
    gini = 0  # 基尼系数初始化
    for lab in unique_labs:  # 遍历各类标签
        n_pos = np.where(labs == lab)[0].shape[0]  # 各标签出现次数
        prob_pos = n_pos / len(labs)  # 各标签出现概率
        gini += prob_pos ** 2  # 计算基尼系数-1
    gini = 1 - gini  # 计算基尼系数-2
    return gini  # 返回基尼系数


def eval_split(datas, labs, ind_fea, split):  # 计算 datas 的 ind_fea 维的基尼增益
    mask = datas[:, ind_fea] <= split  # 是否小于分割阈值
    index_l = np.where(mask == 1)[0]  # 左侧特征索引值(小于分割阈值)
    index_r = np.where(mask == 0)[0]  # 右侧特征索引值(大于分割阈值)
    labs_l = labs[index_l]  # 左侧特征标签
    labs_r = labs[index_r]  # 右侧特征标签

    weight_left = float(len(labs_l) / len(labs))  # 左侧权重
    weight_right = 1 - weight_left  # 右侧权重

    gini_parent = gini_impurity(labs)  # 总基尼系数
    gini_left = gini_impurity(labs_l)  # 左侧基尼系数
    gini_right = gini_impurity(labs_r)  # 右侧基尼系数
    weighted_gini = gini_parent - (weight_left * gini_left + weight_right * gini_right)  # 计算基尼增益

    return weighted_gini  # 返回基尼增益


class node:  # 节点类
    def __init__(self, datas, labs, parent):
        # 处理输入的变量
        self.parent = parent
        self.datas = datas
        self.labs = labs

        # 当前节点的基尼系数
        self.gini = gini_impurity(self.labs)

        # 左右子树初始化
        self.left = None
        self.right = None

        # 当前节点的分割条件
        self.splitting_ind_fea = None  # 待分割的特征
        self.threshold = 0  # 分割阈值

        self.leaf = False  # 叶节点判断 初始化
        self.label = None  # 当前节点标签 初始化
        self.confidence = None  # 叶节点标签的纯度(即该结果的置信度)

    def set_splitting_criteria(self, ind_fea, threshold):  # 设置当前节点的分割条件
        self.splitting_ind_fea = ind_fea
        self.threshold = threshold

    def is_leaf(self, stopping_sz):  # 是否为叶节点(即是否停止分割)
        # 剩下的数据小于stopping_sz/基尼系数为0(即所有数据标签相同)->停止分割
        if len(self.labs) <= stopping_sz or self.gini == 0.0:
            return True
        else:
            return False

    def find_splitting_criterion(self):  # 找到当前节点 最佳的分割维度(ind_fea) 以该维度最佳的分割阈值
        max_score = -1.0  # 最佳基尼系数初始化
        best_ind_fea = None  # 最佳维度初始化
        threshold = 0.0  # 最佳分割阈值初始化

        dim_fea = np.shape(self.datas)[-1]  # 数据集的特征数(即列数)

        for i in range(dim_fea):  # 遍历各特征
            splits = get_possible_splits(self.datas, i)  # 分割阈值 集

            for split in splits:  # 遍历各分割阈值
                split_score =eval_split(self.datas, self.labs, i, split)  # 计算基尼增益
                if split_score > max_score:  # 记录最大的基尼增益
                    max_score = split_score
                    best_ind_fea = i
                    threshold = split

        return max_score, best_ind_fea, threshold  # 返回最大基尼增益、最佳分割维度、最佳分割阈值

    def split(self, ind_fea, threshold):  # 对当前的节点进行分割

        mask = self.datas[:, ind_fea] <= threshold  # 是否小于等于分割阈值

        # 处理输入的变量
        index_l = np.where(mask == 1)[0]  # 左侧特征的索引值
        index_r = np.where(mask == 0)[0]  # 右侧特征的索引值
        labs_l = self.labs[index_l]  # 左侧特征的标签
        labs_r = self.labs[index_r]  # 右侧特征的标签
        datas_l = self.datas[index_l, :]  # 左侧特征
        datas_r = self.datas[index_r, :]  # 右侧特征

        # 输出分割结果
        print("将 %d 分割为 %d 与 %d 特征索引值： %d 分割阈值：%.2f" % (
            len(self.labs), len(labs_l), len(labs_r), ind_fea, threshold))

        # 左右子树
        left = node(datas_l, labs_l, self)
        right = node(datas_r, labs_r, self)

        return left, right  # 返回左右子树

    def set_as_leaf(self):  # 将当前节点设为叶子节点

        self.leaf = True  # 设置该节点为叶节点

        labs = self.labs.tolist()  # 转换为list变量，便于计算
        self.label = max(labs, key=labs.count)  # 设置该节点的标签为数据中数量最多的标签

        n_pos = len(np.where(self.labs == self.label)[0])  # 标签与该节点相同的数量
        self.confidence = float(n_pos / len(self.labs))  # 计算标签纯度


class tree:

    def __init__(self, datas, labs, stopping_sz):

        self.root = None  # 根结点初始化

        # 处理输入的变量
        self.datas = datas
        self.labs = labs
        self.stopping_sz = stopping_sz

        self.dic_tree = {}  # 创建空决策树

    def __build_tree(self, root):

        # 如果是叶子节点则返回
        if root.is_leaf(self.stopping_sz):
            root.set_as_leaf()
            return

        # 如果不是叶子节点，则找最佳分割(基尼增益、最佳特征索引值、最佳分割阈值)
        max_score, best_ind_fea, threshold = root.find_splitting_criterion()

        # 没找到最佳特征索引值则返回
        if best_ind_fea == None:
            return

        # 设置分割条件
        root.set_splitting_criteria(best_ind_fea, threshold)

        # 对当前节点进行分割
        left, right = root.split(best_ind_fea, threshold)
        root.left = left
        root.right = right

        # 递归分割左右子树
        self.__build_tree(root.left)
        self.__build_tree(root.right)
        return

    def fit(self):  # 构建分类树
        if self.root == None:  # 若没有根结点
            self.root = node(self.datas, self.labs, None)  # 创建一个根结点
            self.__build_tree(self.root)  # 创建决策树

    def predict(self, sample):  # 预测标签

        current = self.root
        while (not current.leaf):  # 若当前节点不是叶节点
            # 分割当前节点
            if sample[current.splitting_ind_fea] <= current.threshold:
                current = current.left
            else:
                current = current.right

        return current.label  # 返回当前节点的标签

    def __print_tree(self, root):  # 输出分类树

        if root.leaf:  # 若是叶节点
            return (root.label)  # 返回标签

        ret_Tree = {}  # 分类树
        str_root ='特征维度：%d 分割阈值=%.2f' % (root.splitting_ind_fea, root.threshold)  # 根结点名称

        ret_Tree[str_root] = {}  # 创建空分类树

        # 左右子树名称
        # str_left = "特征维度： %d<分割阈值%.2f"%(root.splitting_ind_fea, root.threshold)
        # str_right = "特征维度： %d>分割阈值%.2f"%(root.splitting_ind_fea, root.threshold)
        str_left = "<%.2f" % (root.threshold)
        str_right = ">%.2f" % (root.threshold)

        ret_Tree[str_root][str_left] = self.__print_tree(root.left)  # 递归输出左子树
        ret_Tree[str_root][str_right] = self.__print_tree(root.right)  # 递归输出右子树

        return ret_Tree  # 返回分类树

    def print_tree(self):  # 输出分类树

        return self.__print_tree(self.root)