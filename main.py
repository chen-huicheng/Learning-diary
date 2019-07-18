#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys                                   #导入必要的包
import numpy as np
from PaintBoard import PaintBoard            #自定义的包
from Recognize import Recognize              #自定义的包
from PyQt5.Qt import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon,QPixmap

class Windows(QWidget):

    def __init__(self):                       # 类的构造函数

        super().__init__()

        self.init()


    def init(self):                           # 初始化函数 被构造函数调用
                            
        self.labels = []                      # 数据初始化 labels为独热码  
        self.images = []                      # images 是图片
        self.paintBoard = PaintBoard()        # 画板
        self.re = Recognize()
                                              # UI初始化
        self.clear = QPushButton('清空',self) # 创建 button 并关联事件处理器函数
        self.clear.setToolTip('点击可清空画板、识别结果、打开的图片')
        self.clear.clicked.connect(self.Clear)

        self.recognize = QPushButton('数字识别',self)
        self.recognize.clicked.connect(self.onClickRecognize)
        
        self.resLabel = QLabel(self)              # 创建 Label 
        self.resLabel.setFixedSize(80,45)
        self.resLabel.setText('识别结果:')
        self.resLabel.setAlignment(Qt.AlignCenter)

        self.result = QLabel(self)
        self.result.setFixedSize(45,45)
        self.result.setFont(QFont('sans-serif',20,QFont.Bold))
        self.result.setStyleSheet('QLabel{border:2px solid black}')
        self.result.setAlignment(Qt.AlignCenter)

        splitter = QSplitter(self)              # 占位符
        
        self.numImage = QLabel(self)
        self.numImage.setAlignment(Qt.AlignCenter)

        self.open = QPushButton('打开图片并识别',self)
        self.open.clicked.connect(self.openImage)

        self.save = QPushButton('保存图片',self)
        self.save.clicked.connect(self.saveImage)

        resLayout = QHBoxLayout()              # 识别结果显示的布局
        resLayout.addWidget(splitter)
        resLayout.addWidget(self.resLabel)
        resLayout.addWidget(self.result)
        resLayout.addWidget(splitter)

        imageLayout = QHBoxLayout()            # 打开图片显示的布局
        imageLayout.addWidget(splitter)
        imageLayout.addWidget(self.numImage)
        imageLayout.addWidget(splitter)

        menu = QVBoxLayout()                   # 创建右侧工具栏  垂直布局
        menu.setContentsMargins(10,10,10,10) 
        menu.addWidget(self.clear)             # 将 button and label 添加到右侧
        menu.addWidget(self.recognize)
        menu.addLayout(resLayout)
        menu.addWidget(splitter)
        menu.addLayout(imageLayout)
        menu.addWidget(self.open)
        menu.addWidget(self.save)

        subLayout1 = QHBoxLayout()  
        subLayout1.addWidget(self.paintBoard)    
        subLayout1.addLayout(menu)

        self.numLabel = QLabel(self)          # 底侧的 button and label
        self.numLabel.setFixedSize(45,30)
        self.numLabel.setText('数字:')
        self.numLabel.setAlignment(Qt.AlignCenter)

        self.number = QSpinBox(self)
        self.number.setFixedSize(35,30)
        self.number.setMaximum(9)
        self.number.setMinimum(0)
        self.number.setSingleStep(1)


        self.addData = QPushButton('添加数据',self)
        self.addData.setToolTip('用于添加数据')
        self.addData.setFixedSize(85,30)
        self.addData.clicked.connect(self.AddData)

        self.learn = QPushButton('执行学习',self)
        self.learn.setFixedSize(85,30)
        self.learn.clicked.connect(self.Learn)

        self.status = QLabel(self)
        self.status.setText('--状态栏--')
        self.status.setToolTip('我是 --状态栏--')
        self.status.setAlignment(Qt.AlignCenter)

        subLayout2 = QHBoxLayout()                  # 底侧的布局
        subLayout2.addWidget(self.numLabel)
        subLayout2.addWidget(self.number)
        subLayout2.addWidget(self.addData)
        subLayout2.addWidget(self.learn)
        subLayout2.addWidget(self.status)

        mainLayout = QVBoxLayout(self)              # 总布局
        mainLayout.setSpacing(10)
        mainLayout.addLayout(subLayout1)
        mainLayout.addLayout(subLayout2)

        self.setLayout(mainLayout) 

        self.setFixedSize(550,340)                  # 设置属性
        self.setWindowTitle('***手写体数字识别***') # 标题
        self.setWindowIcon(QIcon('image/Icon.jpg')) # 图标
        self.center()                               # 居中
        self.show()

                                                    # 事件处理器函数
    def Clear(self):                                # 清空界面
        
        self.paintBoard.Clear()
        self.result.setText('')
        self.numImage.clear()
        self.status.setText('--已清空--')


    def saveImage(self):                            # 保存图片
        
        savePath = QFileDialog.getSaveFileName(self,'saveImage',
                '/home/cheng/test.png','Image(*.png *.jpg)')
        if savePath[0] == '':
            self.status.setText('--取消保存--')
            return
        image = self.paintBoard.GetContentAsQImage()
        print(savePath[0])
        image.save(savePath[0])
        self.status.setText('--图片保存成功--')


    def openImage(self):                           #打开图片 并识别

        openPath = QFileDialog.getOpenFileName(self,'openImage',
                'image','Image(*.png *.jpg *.bmp)')
        if openPath[0] == '':
            self.status.setText('--取消打开图片--')
            return

        print(openPath[0])
        savePath = 'image/test.png'
        self.re.Binarization(openPath[0],savePath)
        image = QPixmap(savePath).scaled(40,40)
        self.numImage.setPixmap(image)

        picture = []
        picture.append(self.re.Picture(savePath))
        label = self.re.recognizeNumber(picture)
        self.result.setText(str(label[0]))
        self.status.setText('--以识别--')


    def onClickRecognize(self):                     # 识别画板图片
        
        image = self.paintBoard.GetContentAsQImage()
        image.save('image/test.png')
        picture = []
        picture.append(self.re.Picture('image/test.png'))
        label = self.re.recognizeNumber(picture)
        self.result.setText(str(label[0]))
        self.status.setText('--以识别--')


    def AddData(self):                              # 添加数据
        
        index = self.number.text()
        label = np.zeros([10])
        label[int(index)] = 1
        self.labels.append(label) 
        image = self.paintBoard.GetContentAsQImage()
        image.save('image/test.png')
        self.images.append(self.re.Picture('image/test.png'))
        self.status.setText('--添加成功--')


    def Learn(self):                                # 执行学习

        if len(self.images) == 0:
            self.status.setText('--没有数据 请添加--')
            return
        
        self.status.setText('--学习中。。。--')
        self.re.Learn(self.images,self.labels)
        self.status.setText('--学习完成--')


    def center(self):                               # 使窗口居中
        
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def closeEvent(self,event):                 # 退出时调用Recognize的析构函数
        del self.re
        event.accept()


if __name__ == '__main__':                      # 只有执行本文件使执行
    
    app = QApplication(sys.argv)
    window = Windows()

    sys.exit(app.exec_())


