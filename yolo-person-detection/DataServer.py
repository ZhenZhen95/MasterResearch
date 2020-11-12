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


class Carame_Accept_Object:
    def __init__(self, S_addr_port=("157.19.105.183", 8880)):
        self.resolution = (640, 480)  # 分辨率
        self.img_fps = 15  # 每秒传输多少帧数
        self.addr_port = S_addr_port
        self.Set_Socket(self.addr_port)

    # 设置套接字
    def Set_Socket(self, S_addr_port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 端口可复用
        self.server.bind(S_addr_port)
        self.server.listen(5)
        # print("the process work in the port:%d" % S_addr_port[1])


def check_option(object, client):
    # 按格式解码，确定帧数和分辨率
    info = struct.unpack('lhh', client.recv(8))
    if info[0] > 888:
        object.img_fps = int(info[0]) - 888  # 获取帧数
        object.resolution = list(object.resolution)
        # 获取分辨率
        object.resolution[0] = info[1]
        object.resolution[1] = info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        return 0


def RT_Image(object, client, D_addr):
    if (check_option(object, client) == 0):
        return
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
    # 设置传送图像格式、帧数
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), object.img_fps]
    while True:
        # print(1)
        # ret, frame = cap.read()  # 读取某一帧
        time.sleep(0.1)  # 推迟线程运行0.1s
        _, object.img = cap.read()
        # if not ret:
        #     print("Done processing !!!")
        #     print("Output file is stored as", output)
        #     cv2.waitKey(3000)
        #     break

        if _ == True:
            # print(2)
            # count = count + 1
            # 抓取为空的帧
            if W is None or H is None:
                (H, W) = object.img.shape[:2]

            # 从输入帧构造一个blob，然后YOLO对象检测器执行前向传递，提供边界框和相关的概率
            blob = cv2.dnn.blobFromImage(object.img, 1 / 255.0, (inpWidth, inpHeight), swapRB=True, crop=False)
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
            isframe = postprocess(object.img, outs)

            end = time.time()

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            # print(isframe)

            # 判断目标
            if isframe:
                object.img = cv2.resize(object.img, object.resolution)  # 按要求调整图像大小(resolution必须为元组)
                _, img_encode = cv2.imencode('.jpg', object.img, img_param)  # 按格式生成图片
                img_code = numpy.array(img_encode)  # 转换成矩阵
                object.img_data = img_code.tostring()  # 生成相应的字符串
                try:
                    # 按照相应的格式进行打包发送图片
                    client.send(
                        struct.pack("lhh", len(object.img_data), object.resolution[0],
                                   object.resolution[1]) + object.img_data)
                except:
                    cap.release()  # 释放资源
                    return

        else:
            break

    # 停止计时器并显示fps信息
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] cleaning up...")

    # cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = Carame_Accept_Object()
    while 1:
        client, D_addr = camera.server.accept()
        clientThread = threading.Thread(None, target=RT_Image, args=(camera, client, D_addr,))
        clientThread.start()
