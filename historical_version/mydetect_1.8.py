# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import ctypes
import math
import os
import platform
import sys
import threading
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

import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, pyqtSignal
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 定义 CameraUI 类
class CameraUI(QtWidgets.QWidget):
    update_frame_signal = pyqtSignal(np.ndarray)  # 定义信号

    def __init__(self):
        super().__init__()
        self.queue = multiprocessing.Queue()  # 初始化队列
        self.shared_value = multiprocessing.Value('i', 0)  # 初始化共享值
        self.shared_value2 = multiprocessing.Value('i', 0)  # 初始化共享值
        self.shared_load = multiprocessing.Manager().list(['', '', '', ''])  # 初始化共享列表
        self.initUI()
        self.detectUI()
        self.show()
        self.detect_process = yolov5_detect()
        self.start_camera_thread()
        self.start_serial_thread()

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

    def start_camera_thread(self):
        self.camera_thread = threading.Thread(target=self.camera_capture)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def start_serial_thread(self):
        self.serial_thread = threading.Thread(target=myserial, args=(self.queue, self.shared_value, self.shared_value2, self.shared_load))
        self.serial_thread.daemon = True
        self.serial_thread.start()

    def camera_capture(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.open_vidadet)
        self.timer.start(20)
        print("摄像头捕获线程启动")

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
            ret, frame = self.detect_process.cap.read()  # 确保这里使用的是正确的视频捕获对象
            if not ret:
                print("无法读取视频帧")
            else:
                print("视频帧读取成功")
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
        if self.detect_process.frame is not None:
            self.update_frame_signal.emit(self.detect_process.frame)

    def update_frame(self, frame):
        print("更新帧")
        frame = cv2.resize(frame, (1080, 600))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(frame.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        p = convertToQtFormat.scaled(1080, 600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(QtGui.QPixmap.fromImage(p))

    def closeEvent(self, event):
        self.timer.stop()
        self.detect_process.cap.release()
        event.accept()

class yolov5_detect():
    def __init__(self, weights='best.pt', device='cpu', conf_thres=0.5, iou_thres=0.45):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
        else:
            print("摄像头打开成功")
        self.retu = {}
        self.classes = ['recyclable waste', 'hazardous waste', 'kitchen waste', 'other waste']
        device = select_device(device)
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = CameraUI()
    ex.update_frame_signal.connect(ex.update_frame)  # 连接信号和槽
    sys.exit(app.exec_())