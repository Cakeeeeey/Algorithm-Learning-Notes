import numpy as np

class functions():
    def __init__(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([[6, 3], [5, 9], [8, 4]])

    def shape_1(self):
        print('a=', self.a)
        print('b=', '\n', self.b)
        print('np.shape(a)=', np.shape(self.a))
        print('np.shape(b)=', np.shape(self.b))
        print('————————1.查看矢量/矩阵尺寸———————————', "\n")

    def type_2(self):
        print('b=', '\n', self.b)
        print('b.dtype=', self.b.dtype)
        print('b.astype(np.float32)=', '\n', self.b.astype(np.float32))
        print('b.astype(np.float32).dtype=', self.b.astype(np.float32).dtype)
        print('————————2.查看/修改数据类型———————————', "\n")

    def array_and_list_3(self):
        print('b=np.array()：', '\n', self.b)
        print('b.tolist()=', self.b.tolist())
        print('list类型可以调用len()函数:len(b)=', len(self.b))
        b = np.array(self.b)  # b的类型由list改回array
        print('————————3.array与list转换、list的len()函数调用———————————', "\n")

    def all_ones_or_zeros_4(self):
        c = np.zeros([3, 2])
        print('np.zeros([3,2])=', '\n', c)
        c = np.ones([3, 2])
        print('np.ones([3,2])', '\n', c)
        print('————————4.创建全0/全1的矩阵———————————', "\n")

    def sum_5(self):
        print('b=', '\n', self.b)
        print('np.sum(b)=', np.sum(self.b))
        print('np.sum(b,axis=1)=', np.sum(self.b, axis=1), 'shape=', np.sum(self.b, axis=1).shape)
        print('np.sum(b,axis=0)=', np.sum(self.b, axis=0), 'shape=', np.sum(self.b, axis=0).shape)
        print('————————5-1.矩阵（沿横/纵维度）求和，结果为矢量———————————', "\n")
        print('np.sum(b,axis=1,keepdims=True)=', '\n', np.sum(self.b, axis=1, keepdims=True), 'shape=',
              np.sum(self.b, axis=1, keepdims=True).shape)
        print('np.sum(b,axis=0,keepdims=True)=', '\n', np.sum(self.b, axis=0, keepdims=True), 'shape=',
              np.sum(self.b, axis=0, keepdims=True).shape)
        print('————————5-2.矩阵（沿横/纵维度）求和，结果为矩阵———————————', "\n")

    def Mean_StandardDeviation_Variance_6(self):
        print('b=', '\n', self.b)
        print('np.mean(b)=', np.mean(self.b))
        print('np.std(b)=', np.std(self.b))
        print('np.var(b)', np.var(self.b))
        print('————————6.矩阵求均值/标准差/方差（沿横/纵维度、结果为矩阵/矢量-函数同上）———————————', "\n")

    def location_of_ArrayElements_7(self):
        print('b=', '\n', self.b)
        print('np.where(b<=3)=', np.where(self.b <= 3), '\n', '#前一个array表示行位置，后一个array表示列位置')
        print('np.where(b<=3)[0].shape[0]=', np.where(self.b <= 3)[0].shape[0])
        print('————————7.查找某元素位置/数量———————————', "\n")

    def sort_and_LocationAfterSort_8(self):
        print('b=', '\n', self.b)
        print('np.sort(-b)=', '\n', np.sort(-self.b))
        print('np.argsort(-b)=', '\n', np.argsort(-self.b), '\n')

        print('np.sort(-b,axis=1)=', '\n', np.sort(-self.b, axis=1))
        print('np.argsort(-b,axis=1)=', '\n', np.argsort(-self.b, axis=1), '\n')

        print('np.sort(b,axis=0)=', '\n', np.sort(self.b, axis=0))
        print('np.argsort(b,axis=0)=', '\n', np.argsort(self.b, axis=0), '\n')

        print('np.sort(-b,axis=0)=', '\n', np.sort(-self.b, axis=0))
        print('\033[1;31m np.argsort(-b,axis=0)= \033[0m', '\n', np.argsort(-self.b, axis=0), '\n')

        print('np.sort(-b,axis=None)=', '\n', np.sort(-self.b, axis=None))
        print('np.argsort(-b,axis=None)=', '\n', np.argsort(-self.b, axis=None))
        print('————————8.（沿横/纵维度）从大到小排序、查找排序后位置———————————', "\n")

    def unique_elements_9(self):
        print('b=', '\n', self.b)
        print('np.unique(b)=', np.unique(self.b))
        print('————————9.获取不重复的元素———————————', "\n")

    def MaxOrMin_and_location_10(self):
        print('b=', '\n', self.b, "\n")

        print('np.max(b)=', np.max(self.b))
        print('np.argmax(b)=', np.argmax(self.b), "\n")

        print('np.max(b,axis=1)=', np.max(self.b, axis=1))
        print('np.argmax(b,axis=1)=', np.argmax(self.b, axis=1), "\n")

        print('np.max(b,axis=0)=', np.max(self.b, axis=0))
        print('np.argmax(b,axis=0)=', np.argmax(self.b, axis=0))
        print('————————10.矩阵（沿横/纵维度）求最大值（最小值max->min）及其位置，'
              '结果为矢量（结果为矩阵则,keepdims=True）———————————', "\n")

    def SampleAveragely_11(self):
        print('np.linspace(0,10,5)=', np.linspace(0, 10, 5))
        print('np.linspace([0,10],[10,20],5)=', '\n', np.linspace([0, 10], [10, 20], 5))
        print('————————11.在区间中均匀采样———————————', "\n")

    def ExpendDims_12(self):
        print('a=', self.a)
        print('a[:,np.newaxis]=', '\n', self.a[:, np.newaxis], 'shape=', self.a[:, np.newaxis].shape)
        print('a[np.newaxis,:]=', self.a[np.newaxis, :], 'shape=', self.a[np.newaxis, :].shape)
        print('————————12-1.维度拓展(np.newaxis方法)———————————', "\n")

        print('a=', self.a)
        print('np.expand_dims(a,axis=1)=', '\n', np.expand_dims(self.a, axis=1), 'shape=', np.expand_dims(self.a, axis=1).shape)
        print('np.expand_dims(a,axis=0)=', np.expand_dims(self.a, axis=0), 'shape=', np.expand_dims(self.a, axis=0).shape)
        print('————————12-2.维度拓展(np.expand_dims方法)———————————', "\n")

    def ConcatanateArrays_13(self):
        print('b=', self.b)
        c = np.ones([3, 2])
        print('c=', c)
        print('np.r_[b,c]=', '\n', np.r_[self.b, c])
        print('np.c_[b,c]=', '\n', np.c_[self.b, c])
        print('————————13-1.两个矩阵拼接(np.r_[]/np.c_[]方法)———————————', "\n")

        print('b=', self.b)
        c = np.ones([3, 2])
        print('c=', c)
        print('np.concatenate((c,b,c),axis=1)=', '\n', np.concatenate((c, self.b, c), axis=1))
        print('np.concatenate((c,b,c),axis=0)=', '\n', np.concatenate((c, self.b, c), axis=0))
        print('————————13-2.多个矩阵拼接(np.concatenate方法)———————————', "\n")

    def DuplicateArray_14(self):
        print('a=', self.a)
        print('np.tile(a,[3,2])=', '\n', np.tile(self.a, [3, 2]))
        print('b=', self.b)
        print('np.tile(b,[3,2])=', '\n', np.tile(self.b, [3, 2]))
        print('————————14.矢量/矩阵的复制———————————', "\n")

    def CalculateBetweenArrayOrVector_15(self):
        print('b=', self.b)
        print('b+b=', '\n', self.b + self.b)
        print('b-b=', '\n', self.b - self.b)
        print('b*b=', '\n', self.b * self.b)
        print('b/b=', '\n', self.b / self.b)
        print('————————15-1.矩阵与矩阵的四则运算———————————', "\n")

        print('b=', self.b)
        print('b+3=', '\n', self.b + 3)
        print('b-3=', '\n', self.b - 3)
        print('b*3=', '\n', self.b * 3)
        print('b/3=', '\n', self.b / 3)
        print('————————15-2.矩阵与标量的四则运算———————————', "\n")

        print('b=', self.b)
        c = 2 * np.ones([3, 1])
        print('c=', c)
        print('b+c=', '\n', self.b + c)
        print('————————15-3.矩阵与列矢量的四则运算———————————', "\n")

        print('b=', self.b)
        c = 2 * np.ones([1, 2])
        print('c=', c)
        print('b+c=', '\n', self.b + c)
        print('————————15-4.矩阵与行矢量的四则运算———————————', "\n")

    def DotBetweenArrayOrVextor_16(self):
        c = np.array([1, 2, 3])
        d = np.array([4, 5, 6])
        print('c=', c)
        print('d=', d)
        print('np.dot(c,d)=', np.dot(c, d))
        print('————————16-1.矢量与矢量（两矢量shape相同）的内积———————————', "\n")

        c = np.array([[1, 1, 7], [3, 4, 3]])
        d = self.b
        print('c=', c)
        print('d=', d)
        print('np.dot(c,d)=', '\n', np.dot(c, d))
        print('————————16-2.矩阵与矩阵(c[M,N] d[N,K])的内积([M,K])———————————', "\n")

        c = np.array([[1, 1, 7], [3, 4, 3]])
        d = np.array([1, 2, 1])
        print('c=', c)
        print('d=', d)
        print('np.dot(c,d)=', '\n', np.dot(c, d))
        print('shape=', np.dot(c, d).shape)
        print('————————16-3.矩阵与矢量(c[M,N] d[N,])的内积([M,])———————————', "\n")

    def PermutationData_17(self):
        print('np.random.permutation(5)=', np.random.permutation(5))
        print('————————17.数据乱序———————————', "\n")

    def RandomAarray_18(self):
        print('np.random.randn(2,1,3)=', np.random.randn(2, 1, 3))
        print('————————18-1.生成正态分布的随机数np.random.randn(组数,行数,列数)———————————', "\n")
        print('np.random.multivariate_normal([2,2],[[.5,0],[0,.5]],10)=', '\n',
              np.random.multivariate_normal([2, 2], [[.5, 0], [0, .5]], 10))
        print('————————18-2.\033[1;31m 生成多维高斯分布的随机数 \033[0m '
              'np.random.multivariate_normal([均值]，[[],[方差]]，数据)———————————', "\n")

    def SaveOrLoad_19(self):
        print('b=', self.b)
        np.save('file_name.npy', self.b)
        print("np.save('file_name.npy',b)")
        e = np.load('file_name.npy')
        print("e=np.load('file_name.npy')=", '\n', e)
        print('————————19-1.array数据的保存/加载———————————', "\n")

        c = [1, 2, 3]
        print('c=', c)
        np.save('file_name.npy', c)
        print("np.save('file_name.npy',c)")
        e = np.load('file_name.npy', allow_pickle=True)
        print("e=np.load('file_name.npy',,allow_pickle=True)=", '\n', e)
        print('————————19-2.list数据的保存/加载(,allow_pickle=True)———————————', "\n")

    def LoadFile_20(self):
        with open("data.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()  # read data.txt in lines

        labs = [line.split(" ")[-1] for line in lines]  # labs=the last column in lines
        labs = np.array(labs).astype(str)  # transform labs to array and its type to str
        print('labs=', labs)

        datas = [line.split(" ")[:1] for line in lines]  # datas=the first column in lines
        datas = np.array(datas).astype(np.float32)  # transform datas to array and its type to float32
        print('datas=', datas)
        print('————————20.读取文件(详见代码)———————————', "\n")

Functions=functions()

Functions.shape_1()
Functions.type_2()
Functions.array_and_list_3()
Functions.all_ones_or_zeros_4()
Functions.sum_5()
Functions.Mean_StandardDeviation_Variance_6()
Functions.location_of_ArrayElements_7()
Functions.sort_and_LocationAfterSort_8()
Functions.unique_elements_9()
Functions.MaxOrMin_and_location_10()
Functions.SampleAveragely_11()
Functions.ExpendDims_12()
Functions.ConcatanateArrays_13()
Functions.DuplicateArray_14()
Functions.CalculateBetweenArrayOrVector_15()
Functions.DotBetweenArrayOrVextor_16()
Functions.PermutationData_17()
Functions.RandomAarray_18()
Functions.SaveOrLoad_19()
Functions.LoadFile_20()