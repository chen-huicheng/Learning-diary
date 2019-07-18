#!/usr/bin/env python3
# -*- coding: utf-8  -*-

import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import *
from PyQt5.QtCore import Qt

class PaintBoard(QWidget):

    def __init__(self):
        super().__init__()

        self.init()                          #初始化

    def init(self):

        self.size = QSize(280,280)

        self.board = QPixmap(self.size)     #新建 QPixmap 作为画板
        self.board.fill(Qt.white)           #用白色填充画板

        self.lastPos = QPoint(0,0)          # 鼠标位置
        self.currentPos = QPoint(0,0)

        self.penThickness = 23              #画笔宽度
        self.penColor = QColor('black')

        self.painter = QPainter()           #定义画笔

        self.setFixedSize(self.size)

    def Clear(self):                        #清空画板，即用白色填充
        self.board.fill(Qt.white)
        self.update()

    def GetContentAsQImage(self):           #将画板转化为图片
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):       #由起始位置到
        self.painter.begin(self)
        self.painter.drawPixmap(0,0,self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):  #鼠标点击

        self.currentPos = mouseEvent.pos()
        self.lastPos = self.currentPos


    def mouseMoveEvent(self,mouseEvent):    #鼠标移动并画线
        self.currentPos = mouseEvent.pos()
        self.painter.begin(self.board)
        self.painter.setPen(QPen(self.penColor,self.penThickness))
        self.painter.drawLine(self.lastPos,self.currentPos)
        self.painter.end()
        self.lastPos = self.currentPos
        self.update()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    pb = PaintBoard()
    pb.show()

    sys.exit(app.exec_())

