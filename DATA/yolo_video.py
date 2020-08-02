import numpy as np
import imutils
from imutils.video import FPS
import time
import cv2

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

# load the COCO class labels our YOLO model was trained on
labelsPath = "yolo/obj.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolo/yolov4.weights"
configPath = "yolo/yolov4.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, pointer to output video file, and
# frame dimensions
# origin_video = "videos/test.mov"
cap = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cap.get(prop))
    print("[INFO] {} total frames in video".format(total))
    time.sleep(2.0)
    fps = FPS().start()

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1


def output(args):
    pass


# scaling_factor = 0.5
# count = 0
# det_num = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Done processing !!!")
        print("Output file is stored as", output)
        cv2.waitKey(3000)
        break

    if ret == True:
        # count = count + 1
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()


        def postprocess(frame, outs):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in outs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > confThreshold:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height

                        # box = detection[0:4] * np.array([W, H, W, H])
                        # (centerX, centerY, width, height) = box.astype("int")

                        centerX = int(detection[0] * frameWidth)
                        centerY = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        left = int(centerX - (width / 2))
                        top = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([left, top, width, height])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('frame', frame)


        # layerOutputs = net.forward(ln)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs)

        end = time.time()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output/videos.avi", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

        # update the FPS counter
        fps.update()

        # write the output frame to disk
        writer.write(frame)

    else:
        break


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] cleaning up...")

cap.release()
writer.release()
cv2.destroyAllWindows()
