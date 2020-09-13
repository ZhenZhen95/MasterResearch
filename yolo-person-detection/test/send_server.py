import cv2
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
ser = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket对象
server.bind(("", 8888))
camera = cv2.VideoCapture(0)
re = server.recvfrom(2048)
print(re, type(re))
object_type = str(re[0], encoding='utf-8')
print(object_type, type(object_type))
while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1000, 700))
    data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 15])[1]
    print(data)
    ser.sendto(data, (re[1][0], 9999))
server.close()
ser.close()
