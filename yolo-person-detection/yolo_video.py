import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "yolo-coco/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelWeights = "yolo-coco/yolov4-tiny.weights"
modelConfiguration = "yolo-coco/yolov4.cfg"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


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

        # def showPicResult(image,peoplecar，outimage):
        #     img = cv2.imread(image)
        #     out_img =outimage
        #     cv2.imwrite(out_img, img)
        #     for i in range(len(peoplecar)):
        #         x1=peoplecar[i][2][0]-peoplecar[i][2][2]/2
        #         y1=peoplecar[i][2][1]-peoplecar[i][2][3]/2
        #         x2=peoplecar[i][2][0]+peoplecar[i][2][2]/2
        #         y2=peoplecar[i][2][1]+peoplecar[i][2][3]/2
        #         im = cv2.imread(out_img)
        #         cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,0),3)
        #         text = listpeoplecar[i][0]
        #         if(text=="people"):
        #             carcol=(55, 55, 255)
        #         else:
        #             carcol = (255, 55, 55)
        #         cv2.putText(im, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.8, carcol, 1, cv2.LINE_AA)
        #         #This is a method that works well.
        #         cv2.imwrite(out_img, im)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "output/yolo_out_py.avi"
if args.image:
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif args.video:
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
# else :
#     filenames = os.listdir(picDir)
#     i = 0
#     num = 0
#     people_num = 0
#
#     people = "people"
#
#     for name in filenames:
#         filename = os.path.join(picDir, name)
#         # print(filename)
#         listpeoplecar = detect(net, meta, filename)
#         print(listpeoplecar)
#         i = i + 1
#         # save_picpath = out_img+str(filename).split("/")[-1].split(".")[0] + ".png"
#         out_img = out_img1 + str(i) + '.png'
#         showPicResult(filename, listpeoplecar，out_img)
#
#         for item in listpeoplecar:
#             # print(item)
#             car_num = car_num + item[0].count(car)  # car个数
#             people_num = people_num + item[0].count(people)  # people个数
#             num = num + 1  # 目标个数
#
#     print('people个数: ' + str(people_num))
#     print('共检测出目标个数: ' + str(num))
#     print('共检测照片个数:' + str(i))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

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

    # Write the frame with the detection boxes
    if args.image:
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))
    cv.imshow(winName, frame)
