import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6 import uic

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

    #基础绘制（按给出的点画直线）
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
        ax2=plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=2) #3x3分割主窗口，在（1，0）位置放置ax2，大小为两行两列
        ax3=plt.subplot2grid((3,3),(1,2),rowspan=2) #3x3分割主窗口，在（1，2）位置放置ax3，大小为两行一列



app = QApplication([])
Functions=functions() #类的实例化
Functions.ui.show()  #显示UI窗口
app.exec()