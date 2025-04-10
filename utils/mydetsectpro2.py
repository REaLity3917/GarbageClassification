# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import argparse
import ctypes
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
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QIcon
from PyQt5.QtCore import QTimer, QSize
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
        self.tim_count = 0

        #串口
        self.queue = multiprocessing.Queue()
        self.shared_value = multiprocessing.Value('i', 0)
        # 创建Manager对象用于创建共享变量

        self.initUI()
        self.detectUI()
        self.videoUI()
        self.show()

        # 机器识别
        self.detect_process = yolov5_detect()
        if not self.detect_process.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头", QMessageBox.Ok)
            self.close()

        self.video = cv2.VideoCapture('/home/liu/Desktop/yolov5-7.0/video.mp4')
        if not self.video.isOpened():
            QMessageBox.critical(self, "错误", "无法加载视频", QMessageBox.Ok)
            self.close()

        #串口
        ser_proeww = multiprocessing.Process(target=myserial, args=(self.queue, self.shared_value,))

        self.timer = QTimer()
        self.timer.timeout.connect(self.open_vidadet)
        self.timer.start(20)
        

    def initUI(self):
        self.setWindowTitle("智能分类垃圾桶")
        self.setGeometry(0, 0, 1080, 600)
        pixmap = QPixmap('/home/liu/Desktop/yolov5-7.0/窗口背景.png')
        # 创建调色板
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pixmap))
        self.setPalette(palette)
    def detectUI(self):
        self.labeld = QLabel(self)
        self.labeld.setGeometry(220, 0, 640, 480)
        self.labeld.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")
        #满载区域
        self.title_label = QLabel(self)
        self.title_label.setText("智 能 分 类 垃 圾 桶")
        self.title_label.setGeometry(300, 500, 500, 50)
        # 设置样式
        self.title_label.setStyleSheet("font-size: 40px; font-weight: bold; border: 2px solid #d3d3d3; color: white;")

        self.labela = QLabel('四个垃圾桶满载情况', self)
        self.labela.setGeometry(870, 0, 210, 50)
        self.labela.setFont(QFont("Arial", 14))
        self.labela.setStyleSheet("color: yellow;")

        self.label0 = QLabel('0%', self)
        self.label0.setGeometry(970, 50, 100, 30)
        self.label0.setFont(QFont("Arial", 10))
        self.label0.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label1 = QLabel('0%', self)
        self.label1.setGeometry(970, 100, 100, 30)
        self.label1.setFont(QFont("Arial", 10))
        self.label1.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label2 = QLabel('0%', self)
        self.label2.setFont(QFont("Arial", 10))
        self.label2.setGeometry(970, 150, 100, 30)
        self.label2.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.label3 = QLabel('0%', self)
        self.label3.setFont(QFont("Arial", 10))
        self.label3.setGeometry(970, 200, 100, 30)
        self.label3.setStyleSheet("color: green;"
                                  "background-color: white;")
        self.btn1 = QLabel('其他垃圾:', self)
        self.btn1.setFont(QFont("Arial", 10))
        self.btn1.setStyleSheet("color: white;")
        self.btn1.setGeometry(870, 50, 100, 30)

        self.btn2 = QLabel('有害垃圾:', self)
        self.btn2.setFont(QFont("Arial", 10))
        self.btn2.setStyleSheet("color: white;")
        self.btn2.setGeometry(870, 100, 100, 30)

        self.btn3 = QLabel('厨余垃圾:', self)
        self.btn3.setFont(QFont("Arial", 10))
        self.btn3.setStyleSheet("color: white;")
        self.btn3.setGeometry(870, 150, 100, 30)

        self.btn4 = QLabel('可回收垃圾:', self)
        self.btn4.setFont(QFont("Arial", 10))
        self.btn4.setStyleSheet("color: white;")
        self.btn4.setGeometry(870, 200, 100, 30)
        #检测信息区域
        self.reportlabel = QLabel(self)
        self.reportlabel.setText("检测信息: ")
        self.reportlabel.setGeometry(0, 0, 150, 50)
        self.reportlabel.setFont(QFont("Arial", 16))
        self.reportlabel.setStyleSheet("color: white;font-weight: bold;")
        self.report = QLabel(self)
        self.report.setGeometry(0, 50, 220, 435)
        self.report.setFont(QFont("Arial", 13))
        self.report.setAlignment(QtCore.Qt.AlignTop)
        self.report.setStyleSheet("color: red;"  # 设置文本显示区域样式
                                  "background-color: white;"
                                  "border-style: outset;"
                                  "border-width: 5;"
                                  "border-radius: 0;"
                                  "border-color: black;")

    def videoUI(self):
        # 显示区域
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1080, 600)
        self.label.setStyleSheet("border-width: 10;"
                                 "border-color: white;"
                                 "background-color: black;")


    def open_vidadet(self):
        if self.tim_count%30==0 or self.detect_process.retu or self.detcount:
            self.detect_process.run()
            self.timer.setInterval(1)
            self.tim_count = 0
        else:
            self.timer.setInterval(20)
        self.tim_count += 1
        if self.detcount >0:
            self.detcount-=1
        if not self.detect_process.retu and not self.detcount:
            ret, frame = self.video.read()
            self.detsign = False
            self.label.show()
        else:
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
            self.detectinfor() if self.infor_sign else "pass"
            self.dp_infor()

    def detectinfor(self):
        count = 1
        # 识别到的垃圾种类，序号，状态，中心坐标
        self.waste_type = [''] * 10
        self.waste_num = []
        self.waste_statu = [''] * 10
        self.waste_address = np.empty(10, dtype=object)

        for mkey,mvalue in self.detect_process.retu.items():

            if mkey[0]=='e':
                type = '其他垃圾'
            elif mkey[0]=='h':
                type = '有害垃圾'
            elif mkey[0] == 'k':
                type = '厨余垃圾'
            elif mkey[0] == 'r':
                type = '可回收垃圾'
            else:
                pass
            self.waste_num.append(count)
            self.waste_type[count] = type
            self.waste_statu[count] = '待处理'
            self.waste_address[count] = [(mvalue[0]+mvalue[2])//2,(mvalue[1]+mvalue[3])//2]
            count +=1
        self.lock_count.append(count) if self.detect_process.retu else self.lock_count.clear()

        if len(self.lock_count)>20:
            self.infor_sign = False if max(self.lock_count)==min(self.lock_count) else True
            self.lock_count.clear()
    def dp_infor(self):
        if self.infor_sign:
            self.reportext = ['']*10
            for i in self.waste_num:
                self.reportext[i-1]=f'{i}--'+self.waste_type[i] +'--'+self.waste_statu[i]
        else:
            self.queue.put(self.detect_process.retu)

        self.report.setText(self.reportext[0]+'\n'+self.reportext[1]+'\n'+self.reportext[2]+'\n')

def myserial(queue,shared_value):
    ser = serial.Serial('ttyUSB0', 9600, timeout=1)# 打开串口
    #串口打开成功
    print('串口打开成功')
    statue = True
    count ,coord= 0,[]
    try:
        while True:
            for i in range(3):
                data = queue.get()
                waste_type,waste_coor = list(data.key()),list(data.value())
                for type,(x0,y0,x1,y1) in waste_type,waste_coor:
                    x ,y = str((x0 + x1) // 2),str((y0 + y1) // 2)
                    x = '00' + str(x) if len(str(x)) == 1 else x
                    y = '00' + str(y) if len(str(y)) == 1 else y
                    x = '0' + str(x) if len(str(x)) == 2 else x
                    y = '0' + str(y) if len(str(y)) == 2 else y
                    co = x+y
                    coord.append(co)

            for t,c in waste_type,coord:
                ser.write(t.encode())
                time.sleep(0.1)
                ser.write(c.encode())
                time.sleep(0.1)
            ser.write(b'o')
            while True:
                res = ser.read()
                res = res.decode()
                if res == '1':
                    print('发送一次成功')
                    break
                elif res == '2':
                    print('发送失败，准备重发')
                    break              # 在此处执行其他操作
    finally:
        # 关闭串口
        if ser.is_open:
            ser.close()

class yolov5_detect():
    def __init__(self ,weights='/home/liu/Desktop/yolov5-7.0/yolov5s.pt', device='0', conf_thres=0.8, iou_thres=1.0):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.retu = {}
        self.classes = ['else waste', 'hazardous', 'kitchen', 'recyclable']
        device = select_device(device=device)
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='/home/liu/Desktop/yolov5-7.0/data/coco128.yaml', fp16=False)
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
        self.pred = non_max_suppression(self.pre, self.conf_thres, self.iou_thres, None, False, max_det=10)
        if self.pred == None:
            t = time.time() - t
            return
        for det in self.pred[0]:
            count += 1
            if count > 1:
                for [a, b] in C:
                    q = abs(int(det[0]) - a)
                    w = abs(int(det[1]) - b)
                    if q <= 5 and w <= 5:
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
        print('retu:',self.retu)
        t = time.time() - t


if __name__=='__main__':

    app = QApplication(sys.argv)
    ex = CameraUI()
    sys.exit(app.exec_())

