import sys

import numpy as np
import imutils
from imutils.video import FPS
import time
import cv2
import numpy
import socket
import threading
import struct

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

labelsPath = "yolo-coco/person/person.names"
LABELS = open(labelsPath).read().strip().split("\n")

# 初始化颜色表示类
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = "yolo-coco/person/person.weights"
configPath = "yolo-coco/person/yolov4-tiny.cfg"

# 加载数据集上训练的yolo对象检测
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# 默认GPU，没有自动切换CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def output(args):
    pass


def SendVideo():
    # 建立sock连接
    # address要连接的服务器IP地址和端口号
    address = ('157.19.105.183', 8888)
    try:
        # 建立socket对象
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)
    # 确定视频文件中的总帧数
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} total frames in video".format(total))
        time.sleep(2.0)
        fps = FPS().start()
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # 读取一帧图像，读取成功:ret=1 frame=读取到的一帧图像；读取失败:ret=0
    ret, frame = cap.read()
    # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
    while ret:
        # print(1)
        # ret, frame = cap.read()  # 读取某一帧
        time.sleep(0.01)  # 推迟线程运行0.1s

        if ret == True:
            # print(2)
            # count = count + 1
            # 抓取为空的帧
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # 从输入帧构造一个blob，然后YOLO对象检测器执行前向传递，提供边界框和相关的概率
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()

            def postprocess(frame, outs):
                isframe = False

                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]

                # 初始化检测边框，置信度和类的列表
                boxes = []
                confidences = []
                classIDs = []

                # 遍历每个图层的输出
                for output in outs:
                    # 遍历每个检测
                    for detection in output:
                        # 提取当前物体检测的类别ID和置信度
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # 检测到的概率大于最小概率 过滤弱预测
                        if confidence > confThreshold:
                            # box = detection[0:4] * np.array([W, H, W, H])
                            # (centerX, centerY, width, height) = box.astype("int")

                            centerX = int(detection[0] * frameWidth)
                            centerY = int(detection[1] * frameHeight)
                            width = int(detection[2] * frameWidth)
                            height = int(detection[3] * frameHeight)

                            left = int(centerX - (width / 2))
                            top = int(centerY - (height / 2))

                            # 更新边界框坐标、置信度、类
                            boxes.append([left, top, width, height])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # 非最大抑制来抑制弱重叠边界框
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

                if len(idxs) > 0:
                    # 循环索引
                    for i in idxs.flatten():
                        # 边框坐标
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        if i < 3:
                            # print(on_press(classIn))
                            print(LABELS[classIDs[i]])
                            # 绘制边框和标签
                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            label = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.imshow('frame', frame)
                            isframe = True
                return isframe

            # layerOutputs = net.forward(ln)
            outs = net.forward(getOutputsNames(net))
            isframe = postprocess(frame, outs)

            end = time.time()

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            # print(isframe)

            # 判断目标
            if isframe:
                print("开始传输...")
                result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                # 建立矩阵
                data = numpy.array(imgencode)
                # 将numpy矩阵转换成字符形式，以便在网络中传输
                stringData = data.tostring()
                # 先发送要发送的数据的长度
                # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
                sock.send(str.encode(str(len(stringData)).ljust(16)))
                # 发送数据
                sock.send(stringData)
                # 读取服务器返回值
                # receive = sock.recv(1024)
                # if len(receive):
                #     print(str(receive, encoding='utf-8'))
                # 读取下一帧图片
                ret, frame = cap.read()
                if cv2.waitKey(10) == 27:
                    break

        else:
            break

    # 停止计时器并显示fps信息
    fps.stop()
    # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # print("[INFO] cleaning up...")

    cap.release()
    writer.release()
    sock.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    SendVideo()
