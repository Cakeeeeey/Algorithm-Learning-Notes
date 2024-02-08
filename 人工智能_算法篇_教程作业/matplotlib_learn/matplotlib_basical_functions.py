import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import uic
import cv2

#中文防乱码
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 切换为图形界面显示的终端TkAgg，防止报错has no attribute ‘FigureCanvas‘
matplotlib.use('TkAgg')

#matplotlib库的各项基本功能
class functions():
    # 加载UI窗口
    def __init__(self):
        # 从文件中加载UI定义
        self.ui = uic.loadUi("UI/Matplotlib Basical Functions.ui")
        self.ui.Button_PaintLine.clicked.connect(self.PaintLine)
        self.ui.Button_MultiPictures.clicked.connect(self.MultiPictures)
        self.ui.Button_IrregulationPictures.clicked.connect(self.IrregulationPictures)
        self.ui.Button_DrawPlot.clicked.connect(self.DrawPlot)
        self.ui.Button_DrawHist.clicked.connect(self.DrawHist)
        self.ui.Button_DrawScatter.clicked.connect(self.DrawScatter)
        self.ui.Button_Legend.clicked.connect(self.Legend)
        self.ui.Button_plt.clicked.connect(self.TitleXYLabelXYLim_plt)
        self.ui.Button_ax.clicked.connect(self.TitleXYLabelXYLim_ax)
        self.ui.Button_ShowPictures.clicked.connect(self.ShowPictures)


    #单屏绘制
    def PaintLine(self):
        data=np.array([1,2,3])  #各点纵坐标（横坐标不规定，默认为0，1，2...）
        plt.plot(data)
        plt.show()
        '''默认
        fig=plt.figure(1) #主窗口
        ax=fig.add_subplot(1,1,1) #1x1副窗口中的窗口1
        ax.plot(data)  #在窗口1上绘制data'''

    #分屏绘制
    def MultiPictures(self):
        fig=plt.figure('Pictures')  #main window

        ax1=fig.add_subplot(1,3,1)  #sub window1(namely ax1) in 1x3 windows
        ax1.plot(np.array(1)) #draw in ax1

        ax2=fig.add_subplot(1,3,2)
        ax2.plot(np.array(2))

        ax3=fig.add_subplot(1,3,3)
        ax3.plot(np.array(3))

        plt.show()

    #不规则分屏绘制
    def IrregulationPictures(self):
        fig=plt.figure('IrregulationPictures')

        ax1=plt.subplot2grid((3,3),(0,0),colspan=3) #3x3分割主窗口，在（0，0）位置放置ax1，大小为一行三列
        ax1.plot(np.random.randn(50).cumsum(),'k--')   #画折线图

        ax2=plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=2) #3x3分割主窗口，在（1，0）位置放置ax2，大小为两行两列
        ax2.hist(np.random.randn(100),bins=20,color='k',alpha=0.3) #画直方图

        ax3=plt.subplot2grid((3,3),(1,2),rowspan=2) #3x3分割主窗口，在（1，2）位置放置ax3，大小为两行一列
        ax3.scatter(np.arange(30),np.arange(30)+3*np.random.randn(30)) #画散点

        plt.show()

    #画折线图
    def DrawPlot(self):
        data=np.random.randn(50).cumsum()  #Y坐标
        fig=plt.figure('折线图') #主窗口

        #只有Y坐标
        ax1=fig.add_subplot(4,1,1) #主窗口分割为4行1列，ax1是其中第一个窗口
        ax1.plot(data) #只有Y坐标

        #Y坐标、X坐标
        ax2=fig.add_subplot(4,1,2) #主窗口分割为4行1列，ax2是其中第二个窗口
        x=np.linspace(0,1,50)  #0-1中间均匀取样50个数
        ax2.plot(x,data) #X坐标，Y坐标

        #Y坐标，X坐标，线型（[color][marker][line]）
        ax3=fig.add_subplot(4,1,3) #主窗口分割为4行1列，ax3是其中第三个窗口
        ax3.plot(x,data,'rs--') #X坐标，Y坐标，线型（[r红色][s方块][--虚线]）

        #Y坐标，X坐标，更详细的线型
        ax4=fig.add_subplot(4,1,4) #主窗口分割为4行1列，ax4是其中第四个窗口
        ax4.plot(x,data,color=(0,0.5,0),marker='*',linewidth=1,markersize=5)

        plt.show()

    #画柱状图
    def DrawHist(self):
        data=np.random.randn(100)  #Y坐标
        fig=plt.figure('柱状图')  #主窗口

        #整数bin(平分数据值域)
        ax1=fig.add_subplot(2,1,1)
        ax1.hist(data,bins=10,color='r',alpha=0.5) #颜色为红色，透明度为0.5

        #区间bin(bin=[1,3,5]，区间为[1,3)、[3,5))
        ax2=fig.add_subplot(2,1,2)
        m_bin=np.linspace(-3,3,10)  #-3到3中间均匀取样10个数
        ax2.hist(data,bins=m_bin,ec='yellow',fc='k',alpha=0.9) #分隔颜色为黄色，柱颜色为黑色，透明度为0.9

        plt.show()

    #画散点图
    def DrawScatter(self):
        data=np.random.randn(2,50) #取2行50列正态分布的数组
        x=data[0,:] #取data第0行所有列
        y=data[1,:] #取data第1行所有列
        value=np.random.rand(50) #取0-1均匀分布的50元素数组
        fig=plt.figure('散点图') #主窗口

        #简单显示（X坐标，Y坐标，点的尺寸，点的颜色，点的形状）
        ax1=fig.add_subplot(1,3,1)
        ax1.scatter(x,y,s=36,c='c',marker='o')

        #点的尺寸、点的颜色（cmap使用色表viridis）随value变化，透明度为0.3
        ax2=fig.add_subplot(1,3,2)
        sizes=(value*200)
        ax2.scatter(x,y,s=sizes,c=value,marker='o',cmap='viridis',alpha=0.3)

        #显示两组数据
        ax3=fig.add_subplot(1,3,3)
        ax3.scatter(x,y,s=60,c='r',marker='*')

        data=np.random.randn(2,50) #取2行50列正态分布的数组
        x=data[0,:] #取data第0行所有列
        y=data[1,:] #取data第1行所有列
        ax3.scatter(x,y,s=36,c='g',marker='*')

        plt.show()

    #图例(显示多组数据)
    def Legend(self):
        fig=plt.figure('图例')

        #多组折线图plot
        ax1=fig.add_subplot(1,2,1)

        data1=np.random.randn(50).cumsum()
        ax1.plot(data1,'rs--',label='data1',linewidth=1,markersize=4)

        data2 = np.random.randn(50).cumsum()
        ax1.plot(data2,'go-',label='data2',linewidth=1,markersize=5)

        ax1.legend(loc='best') #图例自动放置在空白位置

        #多组散点图scatter
        ax2=fig.add_subplot(1,2,2)

        data=np.random.randn(2,50)
        ax2.scatter(data[0,:],data[1:],s=40,c='c',marker='*',label='data1')

        data = np.random.randn(2, 50)
        ax2.scatter(data[0,:],data[1,:],s=60,c='b',marker='o',label='data2')

        ax2.legend(loc='best')

        plt.show()

    #图名称、坐标轴轴名称、坐标轴轴范围(plt方法)
    def TitleXYLabelXYLim_plt(self):
        data=np.random.randn(50).cumsum()

        #plt方法
        plt.plot(data)
        plt.xlim([0,49])  #X轴值域
        plt.ylim([-10,10])  #Y轴值域
        plt.xlabel('x轴')   #X轴名称
        plt.ylabel('y轴')   #Y轴名称
        plt.title('plt方法')

        plt.show()

    # 图名称、坐标轴轴名称、坐标轴轴范围(ax方法)
    def TitleXYLabelXYLim_ax(self):
        data=np.random.randn(50).cumsum()

        #ax方法
        fig=plt.figure('TitleXYLabelXYLim')
        ax=fig.add_subplot(1,1,1)
        ax.plot(data)
        ax.set_xlim([0,49])  #X轴值域
        ax.set_ylim([-10,10])  #Y轴值域
        ax.set_xlabel('x轴')   #X轴名称
        ax.set_ylabel('y轴')   #Y轴名称
        ax.set_title('ax方法')

        plt.show()

    #显示图片
    def ShowPictures(self):
        fig=plt.figure('图像') #主窗口
        img=cv2.imread('picture.png') #导入图像

        ax1=fig.add_subplot(2,2,1)
        img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #图像为RGB
        ax1.imshow(img1) #显示彩图

        ax2=fig.add_subplot(2,2,2)
        img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #图像转为灰度图
        image_gray=ax2.imshow(img2,cmap='gray') #显示灰度图
        plt.colorbar(image_gray,ax=ax2)  #显示色卡

        #显示一个随机矩阵的颜色
        ax3=fig.add_subplot(2,2,3)
        data=np.log(np.random.rand(256,256)) #随机矩阵
        image_random=ax3.imshow(data) #显示随机矩阵
        plt.colorbar(image_random,ax=ax3) #显示色卡

        plt.show()



app = QApplication([])
Functions=functions() #类的实例化
Functions.ui.show()  #显示UI窗口
app.exec()