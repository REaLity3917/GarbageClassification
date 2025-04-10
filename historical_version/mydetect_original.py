# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import ctypes
import math
import os
import platform
import sys
from pathlib import Path
import time
import torch
import serial
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout, QTextEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QIcon, QColor
from PyQt5.QtCore import QTimer, QSize , Qt
import numpy as np
import multiprocessing
FILE = Path(__file__).resolve()#获取当前目录(detect.py)的(使用relsove)绝对路径,并将其赋值给变量FILE F:\yolov5-7.0\mydetect.py
ROOT = FILE.parents[0]  # YOLOv5 root directory 获取上一级目录 F:\yolov5-7.0
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative，绝对路径转换为相对路径 F:\yolov5-7.0\mydetect.py
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
#定义了一共ui的类
#black = np.zeros((480, 640, 3), dtype=np.uint8)


class CameraUI(QWidget):
    def __init__(self):
        super().__init__()
        self.detsign = False
        self.detcount = 0

        self.infor_sign = True
        self.lock_count = []
        self.tim_count = 1
        self.stacount = 0
        self.cc = 0
        self.cli = False
        self.txt  = True
        self.txt2 = []
        self.retxt2 = ['']*4
        self.kk = 0
        self.hh = 0
        self.ee = 0
        self.rr = 0
        self.lmouse,self.rmouse = 0,0
        #串口
        self.queue = multiprocessing.Queue()
        self.shared_value = multiprocessing.Value('i', 0)
        self.shared_value2 = multiprocessing.Value('i', 0)
        self.value = True
        manager = multiprocessing.Manager()
        self.shared_load = manager.list(['','', '', ''])

        # 创建Manager对象用于创建共享变量

        self.initUI()

        self.videoUI()
        self.detectUI()
        self.show()

        # 机器识别
        self.detect_process = yolov5_detect()
        if not self.detect_process.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头", QMessageBox.Ok)
            self.close()

        self.video = cv2.VideoCapture('垃圾分类宣传片.mp4')
        if not self.video.isOpened():
            QMessageBox.critical(self, "错误", "无法加载视频", QMessageBox.Ok)
            self.close()

        #串口
        ser_process = multiprocessing.Process(target=myserial, args=(self.queue, self.shared_value, self.shared_value2,self.shared_load,))
        ser_process.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.open_vidadet)
        self.timer.start(20)


    def initUI(self):
        self.setWindowTitle("智能分类垃圾桶")
        self.setGeometry(0, 0, 1080, 600)

    def detectUI(self):
        size10 = 10
        size16 = 16
        size13 = 13
        size12 = 12
        size14 = 14
        self.labeld = QLabel(self)
        self.labeld.setGeometry(220, 50, 640, 480)
        self.labeld.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")
        #满载区域
        self.title_label = QLabel(self)
        self.title_label.setText("智 能 分 类 垃 圾 桶")
        self.title_label.setGeometry(300, 0, 500, 50)
        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3; color: black;")

        self.labela = QLabel('四个垃圾桶满载情况', self)
        self.labela.setGeometry(870, 0, 210, 50)
        self.labela.setFont(QFont("Arial", size12))
        self.labela.setStyleSheet("color: black;"
                                  "font-weight: bold;")

        self.label0 = QLabel('未满', self)
        self.label0.setGeometry(970, 50, 100, 30)
        self.label0.setFont(QFont("Arial", size10))
        self.label0.setStyleSheet("color: green;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 1;"
                                  "border-radius: 0;"
                                  "border-color: black;"
                                  )
        self.label1 = QLabel('未满', self)
        self.label1.setGeometry(970, 100, 100, 30)
        self.label1.setFont(QFont("Arial", size10))
        self.label1.setStyleSheet("color: green;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 1;"
                                  "border-radius: 0;"
                                  "border-color: black;"
                                  )
        self.label2 = QLabel('未满', self)
        self.label2.setFont(QFont("Arial", size10))
        self.label2.setGeometry(970, 150, 100, 30)
        self.label2.setStyleSheet("color: green;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 1;"
                                  "border-radius: 0;"
                                  "border-color: black;"
                                  )
        self.label3 = QLabel('未满', self)
        self.label3.setFont(QFont("Arial", size10))
        self.label3.setGeometry(970, 200, 100, 30)
        self.label3.setStyleSheet("color: green;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 1;"
                                  "border-radius: 0;"
                                  "border-color: black;"
                                  )
        self.btn1 = QLabel('其他垃圾:', self)
        self.btn1.setFont(QFont("Arial", size10))
        self.btn1.setStyleSheet("color: black;")
        self.btn1.setGeometry(870, 50, 100, 30)

        self.btn2 = QLabel('有害垃圾:', self)
        self.btn2.setFont(QFont("Arial", size10))
        self.btn2.setStyleSheet("color: black;")
        self.btn2.setGeometry(870, 100, 100, 30)

        self.btn3 = QLabel('厨余垃圾:', self)
        self.btn3.setFont(QFont("Arial", size10))
        self.btn3.setStyleSheet("color: black;")
        self.btn3.setGeometry(870, 150, 100, 30)

        self.btn4 = QLabel('可回收垃圾:', self)
        self.btn4.setFont(QFont("Arial", size10))
        self.btn4.setStyleSheet("color: black;")
        self.btn4.setGeometry(870, 200, 100, 30)
        #检测信息区域
        self.reportlabel = QLabel(self)
        self.reportlabel.setText("检测信息: ")
        self.reportlabel.setGeometry(0, 0, 150, 50)
        self.reportlabel.setFont(QFont("Arial", size16))
        self.reportlabel.setStyleSheet("color: black;font-weight: bold;")
        self.report = QLabel(self)
        self.report.setGeometry(0, 50, 220, 220)
        self.report.setFont(QFont("Arial", size13))
        self.report.setAlignment(QtCore.Qt.AlignTop)
        self.report.setStyleSheet("color: black;"  # 设置文本显示区域样式 "font-weight: bold;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 2;"
                                  "border-radius: 0;"
                                  "border-color: black;")
        self.reportlabel1 = QLabel(self)
        self.reportlabel1.setText("已处理垃圾种类数量")
        self.reportlabel1.setGeometry(0, 270, 210, 50)
        self.reportlabel1.setFont(QFont("Arial", size14))
        self.reportlabel1.setStyleSheet("color: black;font-weight: bold;")
        self.report1 = QLabel(self)
        self.report1.setGeometry(0, 320, 220,250 )
        self.report1.setFont(QFont("Arial", size13))
        self.report1.setAlignment(QtCore.Qt.AlignTop)
        self.report1.setStyleSheet("color: green;"  # 设置文本显示区域样式 "font-weight: bold;"
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 2;"
                                  "border-radius: 0;"
                                  "border-color: black;")
        self.button1 = QPushButton('关闭程序',self)
        self.button1.setGeometry(900, 400, 100, 40)
        self.button1.setFont(QFont("Arial", size10))
        self.button2 = QPushButton('关闭摄像头',self)
        self.button2.setFont(QFont("Arial", size10))
        self.button2.setGeometry(900, 450, 100, 40)
        self.button3 = QPushButton('清除数据', self)
        self.button3.setFont(QFont("Arial", size10))
        self.button3.setGeometry(900, 500, 100, 40)
        self.button1.clicked.connect(self.close)
        self.button3.clicked.connect(self.button_3)
        self.b2 = 1
        self.button2.clicked.connect(self.button_2)
    def button_2(self):
        if self.b2 == 1:
            self.detsign = False
            self.detcount = 0
            self.button2.setText('开启摄像头')
            self.b2 = 2
        elif self.b2 == 2:
            self.b2 = 1
            self.button2.setText('关闭摄像头')
    def button_3(self):
        self.retxt2 = ['']*4
        self.report1.setText(self.retxt2[0] + self.retxt2[1] + self.retxt2[2] + self.retxt2[3])

    def detect_hS(self,str):

        if str == 'v':
            self.labeld.hide()
            self.title_label.hide()
            self.label1.hide()
            self.reportlabel.hide()
            self.report.hide()
            self.label.show()

            self.reportlabel1.hide()
            self.report1.hide()

            self.labela.hide()
            self.label0.hide()
            self.label1.hide()
            self.label2.hide()
            self.label3.hide()
            self.btn1.hide()
            self.btn2.hide()
            self.btn3.hide()
            self.btn4.hide()
            self.button3.hide()
            self.button2.hide()
            self.button1.hide()
        else:
            self.labela.setStyleSheet("color: black;"
                                      "font-weight: bold;")
            self.btn1.setStyleSheet("color: black;")
            self.btn2.setStyleSheet("color: black;")
            self.btn3.setStyleSheet("color: black;")
            self.btn4.setStyleSheet("color: black;")
            self.reportlabel1.setStyleSheet("color: black;font-weight: bold;")
            self.labeld.show()
            self.title_label.show()
            self.label.hide()
            self.label1.show()
            self.reportlabel.show()
            self.report.show()
            self.reportlabel1.show()
            self.report1.show()
            self.labela.show()

            self.label0.show()
            self.label1.show()
            self.label2.show()
            self.label3.show()
            self.btn1.show()
            self.btn2.show()
            self.btn3.show()
            self.btn4.show()
            self.button3.show()
            self.button2.show()
            self.button1.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # 检查是否为左键点击

            position = event.pos()
            if position.x()>540 and position.y()<320:
                self.lmouse += 1
                if self.lmouse == 1:
                    if not self.labela.isVisible():
                        self.labela.show()
                        self.labela.setStyleSheet("color: white;"
                                                  "font-weight: bold;")
                        self.label0.show()
                        self.label1.show()
                        self.label2.show()
                        self.label3.show()
                        self.btn1.show()
                        self.btn1.setStyleSheet("color: white;")
                        self.btn2.show()
                        self.btn2.setStyleSheet("color: white;")
                        self.btn3.show()
                        self.btn3.setStyleSheet("color: white;")
                        self.btn4.show()
                        self.btn4.setStyleSheet("color: white;")
                    else:
                        self.lmouse = 0


                else:  # 检查是否为右键点击
                    self.labela.hide()
                    self.label0.hide()
                    self.label1.hide()
                    self.label2.hide()
                    self.label3.hide()
                    self.btn1.hide()
                    self.btn2.hide()
                    self.btn3.hide()
                    self.btn4.hide()
                    self.labela.setStyleSheet("color: black;"
                                              "font-weight: bold;")
                    self.btn1.setStyleSheet("color: black;")
                    self.btn2.setStyleSheet("color: black;")
                    self.btn3.setStyleSheet("color: black;")
                    self.btn4.setStyleSheet("color: black;")
                    self.lmouse = 0
            elif  position.x()<540 and position.y()>320:
                self.rmouse += 1
                if self.rmouse == 1:
                    if  not self.reportlabel1.isVisible():
                        self.reportlabel1.show()
                        self.reportlabel1.setStyleSheet("color: white;font-weight: bold;")
                        self.report1.show()
                    else:
                        self.rmouse = 0
                else:
                    self.reportlabel1.hide()
                    self.reportlabel1.setStyleSheet("color: black;font-weight: bold;")
                    self.report1.hide()
                    self.rmouse = 0
            elif position.x() > 540 and position.y() > 320:
                if self.button3.isVisible():
                    self.button3.hide()
                    self.button2.hide()
                    self.button1.hide()
                else:
                    self.button3.show()
                    self.button2.show()
                    self.button1.show()
        # 获取鼠标点击位置
    def videoUI(self):
        # 显示区域

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1080, 600)
        self.label.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")
        self.label.hide()


    def open_vidadet(self):

        if self.tim_count%30==0 or self.detect_process.retu or self.detcount:
            if self.b2 == 1:
                self.detect_process.run()
            else:
                self.detect_process.retu = {}
            self.timer.setInterval(1)
            self.tim_count = 0

        else:
            self.timer.setInterval(20)

        self.tim_count += 1
        if self.detcount >0:
            self.detcount-=1
        if not self.detect_process.retu and not self.detcount:
            ret, frame = self.video.read()
            self.infor_sign = True
            self.detsign = False
            if not self.label.isVisible():
                self.detect_hS('v')

        else:

            if self.label.isVisible():
                self.detect_hS('d')
            ret ,self.detsign = True ,True
            if self.detect_process.retu:
                self.detcount = 100
            frame = self.detect_process.frame
            self.label.hide()

        if not ret:
            # 到达视频结尾，将VideoCapture对象的位置设置为0，即重新播放视频
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        self.up_frame(frame,1080,600) if not self.detsign else self.up_frame(frame, 640, 480)



    def up_frame(self,frame,vw,vh):

        frame = cv2.resize(frame,(vw,vh))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(vw, vh, aspectRatioMode=1, transformMode=0)
        self.label.setPixmap(QPixmap.fromImage(p)) if not self.detsign else self.labeld.setPixmap(QPixmap.fromImage(p))
        if self.detsign:
            if self.infor_sign:
                self.detectinfor()
            else:
                self.detectinfor2()
            self.shared_value2.value = len(self.detect_process.retu)

            self.dp_infor()
        else:
            self.change()

    def detectinfor2(self):
        ee = {}
        rr = {}
        hh = {}
        kk = {}
        for a, b in self.detect_process.retu.items():
            if 'kitchen' in a:
                kk[a] = b
            elif 'hazardous' in a:
                hh[a] = b
            elif 'recyclable waste' in a:
                rr[a] = b
            elif 'other waste' in a:
                ee[a] = b
        self.detect_process.retu = {**rr, **hh, **ee, **kk}
        #print(self.detect_process.retu)

        self.lock_count.append(len(self.detect_process.retu))

        if len(self.lock_count) > 10:
            if max(self.lock_count) == min(self.lock_count):
                self.cli = True
            else:
                self.cli = False
            self.lock_count.clear()
    def detectinfor(self):
        ee={}
        rr={}
        hh={}
        kk={}
        for a,b in self.detect_process.retu.items():
            if 'kitchen' in a:
                kk[a] = b
            elif 'hazardous' in a:
                hh[a]=b
            elif 'recyclable waste' in a:
                rr[a]=b
            elif 'other waste' in a:
                ee[a]=b
        self.detect_process.retu = {**rr,**hh,**ee,**kk}
        #print(self.detect_process.retu)


        self.lock_count.append(len(self.detect_process.retu))
        if self.detect_process.retu:
            if len(self.lock_count)>21:
                if max(self.lock_count)==min(self.lock_count):
                    self.infor_sign = False
                    self.cli = True
                    self.stacount = len(self.detect_process.retu)

                else:
                    self.cli = False
                    self.infor_sign = True
                self.lock_count.clear()
    def dp_infor(self):
        self.e,self.r,self.k,self.h = 0,0,0,0
        if self.infor_sign:

            report = ['']*4
            self.st = 1
            count1 = 0
            for mkey in self.detect_process.retu.keys():
                count1 += 1

                if mkey[0] == 'e':
                    self.e +=1
                    if self.e == 1:
                        report[2] = f'{count1}其他垃圾     1待处理'
                        eee =count1
                    elif self.e == 2:
                        report[2] = report[2] + '\n' +f'{eee}其他垃圾     '+ f'{self.e}待处理'
                    elif self.e == 3:
                        report[2] = report[2] +'\n' + f'{eee}其他垃圾     '+ f'{self.e}待处理'
                    elif self.e == 4:
                        report[2] = report[2] + '\n' + f'{eee}其他垃圾     ' + f'{self.e}待处理'
                elif mkey[0] == 'h':
                    self.h += 1
                    if self.h == 1:
                        report[1] = f'{count1}有害垃圾     1待处理'
                        eewe = count1
                    elif self.h == 2:
                        report[1] = report[1] + '\n' + f'{eewe}有害垃圾     ' + f'{self.h}待处理'
                    elif self.h == 3:
                        report[1] = report[1] + '\n' +f'{eewe}有害垃圾     '+ f'{self.h}待处理'
                elif mkey[0] == 'k':
                    self.k += 1
                    if self.k == 1:
                        report[3] = f'{count1}厨余垃圾     1待处理'
                        kkk = count1
                    elif self.k == 2:
                        report[3] = report[3] +  '\n' +f'{kkk}厨余垃圾     ' + f'{self.k}待处理'
                    elif self.k == 3:
                        report[3] = report[3] + '\n' +f'{kkk}厨余垃圾     '  + f'{self.k}待处理'
                    elif self.k == 4:
                        report[3] = report[3] + '\n' + f'{kkk}厨余垃圾     ' + f'{self.k}待处理'
                elif mkey[0] == 'r':
                    self.r += 1
                    if self.r==1:
                        report[0] = '1可回收垃圾 1待处理'
                    elif self.r==2:
                        report[0] = report[0] +'\n' + '1可回收垃圾 ' + f'{self.r}待处理'
                    elif self.r == 3:
                        report[0] = report[0] + '\n' + '1可回收垃圾 '+ f'{self.r}待处理'
            self.reportext = sorted(report, key=custom_sort)

        else:

            if self.shared_value.value or self.st:
                if self.detect_process.retu:
                    first_key = list(self.detect_process.retu)[0]
                    first_value = self.detect_process.retu[first_key]
                    new_dict = {first_key: first_value}
                    self.queue.put(new_dict)
                    self.st=0
                self.shared_value.value = 0
                self.change_txt()

        self.report.setText(self.reportext[0]+'\n'+self.reportext[1]+'\n'+self.reportext[2]+'\n'+self.reportext[3]+'\n')
        if not self.value:
            self.report.setText('垃圾已全部清理完毕')
            self.report.setStyleSheet("color: green;"  # 设置文本显示区域样式 "font-weight: bold;"
                                      "background-color: white;"
                                      "border-style: outset;"
                                      "border-width: 2;"
                                      "border-radius: 0;"
                                      "border-color: black;")
        if self.detect_process.retu:
            self.value = True
            self.report.setStyleSheet("color: black;"  # 设置文本显示区域样式 "font-weight: bold;"
                                      "background-color: white;"
                                      "border-style: outset;"
                                      "border-width: 2;"
                                      "border-radius: 0;"
                                      "border-color: black;")
        self.change()





    def change(self):
        if self.shared_load[0] == 'e':
            self.label0.setText("已满载")
            self.label0.setStyleSheet("color: red;"
                                      "background-color: white;"
                                       "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )

        else:
            self.label0.setText("未满")
            self.label0.setStyleSheet("color: green;"
                                      "background-color: white;"
                                      "border-style: outset;"
                                      "border-width: 1;"
                                      "border-radius: 0;"
                                      "border-color: black;"
                                      )
        if self.shared_load[1] == 'h':
            self.label1.setText("已满载")
            self.label1.setStyleSheet("color: red;"
                                      "background-color: white;"
                                       "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )

        else:
            self.label1.setText("未满")
            self.label1.setStyleSheet("color: green;"
                                      "background-color: white;"
                                      "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )
        if self.shared_load[3] == 'k':
            self.label2.setText("已满载")
            self.label2.setStyleSheet("color: red;"
                                      "background-color: white;"
                                     "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )
        else:
            self.label2.setText("未满")
            self.label2.setStyleSheet("color: green;"
                                      "background-color: white;"
                                     "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )
        if self.shared_load[2] == 'r':
            self.label3.setText("已满载")
            self.label3.setStyleSheet("color: red;"
                                      "background-color: white;"
                                       "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )
        else:
            self.label3.setText("未满")
            self.label3.setStyleSheet("color: green;"
                                      "background-color: white;"
                                     "border-style: outset;"
                                       "border-width: 1;"
                                       "border-radius: 0;"
                                      "border-color: black;"
                                      )
    def change_txt(self):

        print('self.stacount - self.shared_value2.value',(self.stacount - self.shared_value2.value))

        if  (self.stacount - self.shared_value2.value)==self.stacount:
            self.value = False
            self.infor_sign = True
        elif  (self.stacount - self.shared_value2.value) == 0:
            self.reportext[0] = self.reportext[0].replace('1待处理', '1处理中')

            self.cc =11

        elif  (self.stacount - self.shared_value2.value) == 1:
            self.reportext[0] = self.reportext[0].replace('1处理中', '1-OK')
            if self.detect_process.retu:
                if '2待处理' in self.reportext[0]:
                    self.reportext[0] = self.reportext[0].replace('2待处理', '2处理中')
                    self.cc = 12
                else:
                    self.reportext[1] = self.reportext[1].replace('1待处理', '1处理中')
                    self.cc = 21
        elif  (self.stacount - self.shared_value2.value) == 2:
            if self.cc==12:
                self.reportext[0] = self.reportext[0].replace('2处理中', '2-OK')
                if '3待处理' in self.reportext[0]:
                    self.reportext[0] = self.reportext[0].replace('3待处理', '3处理中')
                    self.cc == 13
                else:
                    self.reportext[1] = self.reportext[1].replace('1待处理', '1处理中')
                    self.cc = 21
            elif self.cc==21:
                self.reportext[1] = self.reportext[1].replace('1处理中', '1-OK')
                if '2待处理' in self.reportext[1]:
                    self.reportext[1] = self.reportext[1].replace('2待处理', '2处理中')
                    self.cc = 22
                else:
                    self.reportext[2] = self.reportext[2].replace('1待处理', '1处理中')
                    self.cc = 31
        elif (self.stacount - self.shared_value2.value) == 3:
            if self.cc == 21:
                self.reportext[1] = self.reportext[1].replace('1处理中', '1-OK')
                if '2待处理' in self.reportext[1]:
                    self.reportext[1] = self.reportext[1].replace('2待处理', '2处理中')
                    self.cc == 22
                else:
                    self.reportext[2] = self.reportext[2].replace('1待处理', '1处理中')
                    self.cc = 31
            elif self.cc == 31:
                self.reportext[2] = self.reportext[2].replace('1处理中', '1-OK')
                if '2待处理' in self.reportext[2]:
                    self.reportext[2] = self.reportext[2].replace('2待处理', '2处理中')
                    self.cc = 32
                else:
                    self.reportext[3] = self.reportext[3].replace('1待处理', '1处理中')
                    self.cc = 41
        elif (self.stacount - self.shared_value2.value) == 4:
            if self.cc == 22:
                self.reportext[1] = self.reportext[1].replace('2处理中', '2-OK')
                if '3待处理' in self.reportext[1]:
                    self.reportext[1] = self.reportext[1].replace('3待处理', '3处理中')
                    self.cc == 23
                else:
                    self.reportext[2] = self.reportext[2].replace('1待处理', '1处理中')
                    self.cc = 31
            elif self.cc == 31:
                self.reportext[2] = self.reportext[2].replace('1处理中', '1-OK')
                if '2待处理' in self.reportext[2]:
                    self.reportext[2] = self.reportext[2].replace('2待处理', '2处理中')
                    self.cc = 32
                else:
                    self.reportext[3] = self.reportext[3].replace('1待处理', '1处理中')
                    self.cc = 41
            elif self.cc == 32:
                self.reportext[2] = self.reportext[2].replace('2处理中', '2-OK')
                if '3待处理' in self.reportext[2]:
                    self.reportext[2] = self.reportext[2].replace('3待处理', '3处理中')
                    self.cc = 33
                else:
                    self.reportext[3] = self.reportext[3].replace('1待处理', '1处理中')
                    self.cc = 41

        mi = 0


        if self.txt:

            self.last_wastes = self.detect_process.retu
            self.txt = False

        if len(self.detect_process.retu) < len(self.last_wastes):
            if (self.stacount - self.shared_value2.value) == self.stacount:
                self.txt = True
            print(self.last_wastes)
            first_key = next(iter(self.last_wastes))
            print(first_key)
            self.txt2.append(first_key)
            print('txt2',self.txt2)
            self.last_wastes = self.detect_process.retu
            for a in self.txt2:
                if 'kitchen' in a:
                    self.kk +=1
                elif 'hazardous' in a:
                    self.hh += 1
                elif 'recyclable waste' in a:
                    self.rr += 1
                elif 'other waste' in a:
                    self.ee += 1
            if self.rr:
                mi +=1
                self.retxt2[0]= f'{mi}可回收垃圾 {self.rr}--OK\n'
            if self.ee:
                mi += 1
                self.retxt2[2]= f'{mi}其他垃圾     {self.ee}--OK\n'
            if self.hh:
                mi += 1
                self.retxt2[1]= f'{mi}有害垃圾     {self.hh}--OK\n'
            if self.kk:
                mi += 1
                self.retxt2[3]= f'{mi}厨余垃圾     {self.kk}--OK\n'
            self.rr,self.ee,self.kk,self.hh = 0,0,0,0
        print('retxt2',self.retxt2)
        self.report1.setText(self.retxt2[0]+self.retxt2[1]+self.retxt2[2]+self.retxt2[3])
def myserial(queue,shared_value,shared_value2,shared_load):
    ser = serial.Serial('COM6', 9600, timeout=1)# 打开串口
    time.sleep(5)
    #串口打开成功
    print('串口打开成功')

    ser_status = 0
    ace = []#平均值
    count ,coord= 0,[]
    try:
        while True:
            if not queue.empty():

                data = queue.get()
                print('data',data)
                waste_type,waste_coor = list(data.keys()),list(data.values())

                for (x0, y0, x1, y1) in  waste_coor:
                    ac = str((x0 + x1 + y0 + y1)//4)
                    x0 = str(x0)
                    y0 = str(y0)
                    x1 = str(x1)
                    y1 = str(y1)

                    if len(x0) < 3:
                        x0 = '0' * (3 - len(x0)) + x0
                    if len(x1) < 3:
                        x1 = '0' * (3 - len(x1)) + x1
                    if len(y0) < 3:
                        y0 = '0' * (3 - len(y0)) + y0
                    if len(y1) < 3:
                        y1 = '0' * (3 - len(y1)) + y1
                    if len(ac) < 3:
                        ac = '0' * (3 - len(ac)) + ac
                    co = x0+y0+x1+y1
                    ace.append(ac)
                    print('ac:',ac)
                    coord.append(co)
                print("coord:",coord)

                while True:
                    ser.write(str(shared_value2.value).encode())
                    print(shared_value2.value)
                    time.sleep(0.1)
                    ser.write(waste_type[0][0].encode())
                    print('waste_type:',waste_type[0][0])
                    time.sleep(0.1)
                    ser.write(coord[0].encode())
                    time.sleep(0.1)
                    ser.write(ace[0].encode())
                    time.sleep(0.1)

                    print('发送一次成功')
                    ser.write(b'o')

                    while True:
                        T = time.time()
                        res = ser.read()
                        if res:
                            res = res.decode()
                            #print("res", res)
                            if res == '1':
                                print('成功')

                            elif res == '2':
                                ser_status = 1
                                shared_value.value = 1

                                break
                            elif res == '0':
                                print('发送错误,重发')
                                break
                            elif res == 'e':
                                shared_load[0] = 'e'
                            elif res == 'h':
                                shared_load[1] = 'h'
                            elif res == 'r':
                                shared_load[2] = 'r'
                            elif res == 'k':
                                shared_load[3] = 'k'
                            elif res == 'c':
                                shared_load[0] = ''
                            elif res == 'b':
                                shared_load[1] = ''
                            elif res == 'a':
                                shared_load[2] = ''
                            elif res == 'd':
                                shared_load[3] = ''


                        T=time.time()-T
                        if T>2:
                            break
                    print('发送下一个垃圾')
                    if ser_status:
                        ser_status = 0
                        break
                ace.clear()
                data.clear()
                coord.clear()
            else:
                res_=ser.read()
                res_ = res_.decode()

                if res_ == 'e':
                    shared_load[0] = 'e'
                elif res_ == 'h':
                    shared_load[1] = 'h'
                elif res_ == 'r':
                    shared_load[2] = 'r'
                elif res_ == 'k':
                    shared_load[3] = 'k'
                elif res_ == 'c':
                    shared_load[0] = ''
                elif res_ == 'b':
                    shared_load[1] = ''
                elif res_ == 'a':
                    shared_load[2] = ''
                elif res_ == 'd':
                    shared_load[3] = ''
                #rint(res_)
    finally:
        # 关闭串口
        if ser.is_open:
            ser.close()

class yolov5_detect():
    def __init__(self ,weights='best.pt', device='0', conf_thres=0.25, iou_thres=0.45):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.retu = {}
        # self.classes = ['else waste', 'hazardous', 'kitchen', 'recyclable']
        self.classes = ['recyclable waste', 'hazardous waste', 'kitchen waste', 'other waste']
        device = select_device(device=device)
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='data/waste-classification.yaml', fp16=False)
        self.device = device
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
    def run(self):

        self.retu = {}
        count, C = 0, []
        lase_type = []
        t = time.time()
        self.ret, self.frame = self.cap.read()

        self.frame = cv2.resize(self.frame, (800, 600))
        self.frame = self.frame[(600 // 2 - 240):(600 // 2 + 240), (800 // 2 - 320):(800 // 2 + 320)]

        self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.img = torch.from_numpy(self.img).to(self.device).float() / 255.0
        self.img = self.img.permute(2, 0, 1).unsqueeze(0)
        if len(self.img.shape) == 3:
            self.img = self.img[None]  # expand for batch dim如果张量的维度为 3，则在第 0 维上添加一个维度，以便将其扩展为批次大小为 1 的张量。
        self.pre = self.model.model(self.img, augment=False)  # 调用 YOLOv5 模型的 model 方法，对输入的图像或视频进行推理，并得到目标检测结果。
        self.pred = non_max_suppression(self.pre, self.conf_thres, self.iou_thres, None, False, max_det=20)
        if self.pred == None:
            t = time.time() - t
            return
        for det in self.pred[0]:
            count += 1
            if count > 1:
                for [a, b] in C:
                    q = abs(int(det[0]) - a)
                    w = abs(int(det[1]) - b)
                    if q <= 15 and w <= 15:
                        ab = False
                        break
                    ab = True
            else:
                ab = True
            if ab:
                xyxy = (det[0], det[1], det[2], det[3])
                cls = det[5]
                conf = det[4]
                ty = self.classes[int(cls)] + str(count) if int(det[5]) in lase_type else self.classes[int(cls)]
                self.retu[ty] = list(int(x) for x in xyxy)
                label = f'{self.classes[int(cls)]} {conf:.2f}'
                cv2.rectangle(self.frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(self.frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                lase_type.append(int(det[5]))
            C.append([int(det[0]), int(det[1])])
        #print('retu:',self.retu)
        t = time.time() - t

def custom_sort(x):
    if x != '':
        return 0  # 将空字符串映射为一个很大的数值
    else:
        return 1

if __name__=='__main__':

    app = QApplication(sys.argv)
    ex = CameraUI()
    sys.exit(app.exec_())

