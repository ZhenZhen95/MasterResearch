import threading

import cv2
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import socket

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image


server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
ser = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
server.bind(("", 8888))
re = None
object_type = "people"

# server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
# ser = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
# server.bind(("", 8888))
# re = server.recvfrom(2048)
# print(re, type(re))
# object_type = str(re[0], encoding='utf-8')
# print(object_type, type(object_type))

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()


# Load names of classes
# classesFile = "yolo-coco/person/person.names"
# classes = None
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')
#
# # Give the configuration and weight files for the model and load the network using them.
# modelWeights = "yolo-coco/person/person.weights"
# modelConfiguration = "yolo-coco/person/yolov4-tiny.cfg"
#
# net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers,the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using nms
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:  # 遍历每的图层的输出
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:  # 过滤
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                print(classId, confidence, boxes)

    # NMS消除置信度底的冗余叠框
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if classId == 0:
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


def receive_data():
    global re, object_type
    re = server.recvfrom(2048)
    print(re, type(re))
    object_type = str(re[0], encoding='utf-8')
    print(object_type, type(object_type))


def receive_data_continuous():
    while True:
        receive_data()


# Process inputs
winName = 'object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# outputFile = "output/yolo_out_py0.avi"
# if args.video:
#     # Open the video file
#     if not os.path.isfile(args.video):
#         print("Input video file ", args.video, " doesn't exist")
#         sys.exit(1)
#     cap = cv.VideoCapture(args.video)
#     outputFile = args.video[:-4] + '_yolo_out_py.avi'
# else:
#     # Webcam input
#     cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
# if (not args.image):
#     vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
#                                 (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# def main_process(classes, frame):
receive_data()
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('videos/test.mov')
threading.Thread(target=receive_data_continuous).start()
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    if object_type == 'people':
        classesFile = "yolo-coco/person/person.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelWeights = "yolo-coco/person/person.weights"
        modelConfiguration = "yolo-coco/person/yolov4-tiny.cfg"

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        # Put efficiency information.
        # The function getPerfProfile returns the overall time for inference(t)
        # and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        frame = cv2.resize(frame, (1000, 700))
        data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
        print(data)
        ser.sendto(data, (re[1][0], 9999))

        # Write the frame with the detection boxes
        # if args.image:
        #     cv.imwrite(outputFile, frame.astype(np.uint8))
        # else:
        #     vid_writer.write(frame.astype(np.uint8))
        cv.imshow(winName, frame)

    if object_type == 'bus':
        classesFile = "yolo-coco/bus/bus.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelWeights = "yolo-coco/bus/bus.weights"
        modelConfiguration = "yolo-coco/yolov4-tiny.cfg"

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        # Put efficiency information.
        # The function getPerfProfile returns the overall time for inference(t)
        # and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        frame = cv2.resize(frame, (1000, 700))
        data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
        # print(data)
        ser.sendto(data, (re[1][0], 9999))

        cv.imshow(winName, frame)

    if object_type == 'car':
        classesFile = "yolo-coco/car/car.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelWeights = "yolo-coco/car/car.weights"
        modelConfiguration = "yolo-coco/yolov4-tiny.cfg"

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        # Put efficiency information.
        # The function getPerfProfile returns the overall time for inference(t)
        # and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        frame = cv2.resize(frame, (1000, 700))
        data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
        # print(data)
        ser.sendto(data, (re[1][0], 9999))

        cv.imshow(winName, frame)

    if object_type == 'chair':
        classesFile = "yolo-coco/chair/chair.names"
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        modelWeights = "yolo-coco/chair/chair.weights"
        modelConfiguration = "yolo-coco/yolov4-tiny.cfg"

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        # Put efficiency information.
        # The function getPerfProfile returns the overall time for inference(t)
        # and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        # print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        frame = cv2.resize(frame, (1000, 700))
        data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
        # print(data)
        ser.sendto(data, (re[1][0], 9999))

        # Write the frame with the detection boxes
        # if args.image:
        #     cv.imwrite(outputFile, frame.astype(np.uint8))
        # else:
        #     vid_writer.write(frame.astype(np.uint8))
        cv.imshow(winName, frame)

    if object_type == 'table':
            classesFile = "yolo-coco/tvmonitor/tvmonitor.names"
            classes = None
            with open(classesFile, 'rt') as f:
                classes = f.read().rstrip('\n').split('\n')

            # Give the configuration and weight files for the model and load the network using them.
            modelWeights = "yolo-coco/tvmonitor/tvmonitor.weights"
            modelConfiguration = "yolo-coco/yolov4-tiny.cfg"

            net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            postprocess(frame, outs)

            # Put efficiency information.
            # The function getPerfProfile returns the overall time for inference(t)
            # and the timings for each of the layers(in layersTimes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            # print(label)
            cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            frame = cv2.resize(frame, (1000, 700))
            data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
            # print(data)
            ser.sendto(data, (re[1][0], 9999))

            # Write the frame with the detection boxes
            # if args.image:
            #     cv.imwrite(outputFile, frame.astype(np.uint8))
            # else:
            #     vid_writer.write(frame.astype(np.uint8))
            cv.imshow(winName, frame)
server.close()
ser.close()

