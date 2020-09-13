from tkinter import Tk, Frame
# import keyboard
from pynput import keyboard
import os
import queue
import threading
import time

# root = Tk()

# def key(event):
#     print("pressed", repr(event.char))
#
#
# # def callback(event):
# #     print("clicked at", event.x, event.y)
#
#
# frame = Frame(root, width=100, height=100)
# frame.bind("<Key>", key)
# # frame.bind("<Button-1>", callback)
# frame.pack()
#
# root.mainloop()

# classIn = 0
#
#
# def on_press(key):
#     global classIn
#     # print(str(key))
#     if str(key) == 'Key.ctrl_l':
#         # classIn = int(input("input:"))
#         classIn = int(float(input('input:')))
#         if classIn == 1:
#             print(classIn)
#             os.system("yolo_video_chair.py &")
#
#         elif classIn == 2:
#             print(classIn)
#             # os.system("yolo_video_person.py &")
#
#         elif classIn == 3:
#             print(classIn)
#             # os.system("yolo_video_tvmonitor.py &")
#
#
# def on_release(key):
#     print('{0} released'.format(key))
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False
#
#
# keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# lst = [keyboard_listener]
#
# for t in lst:
#     t.start()
# for t in lst:
#     t.join()

# ---------------------------------------------------------------------

# exitFlag = 0
#
#
# class myThread(threading.Thread):
#     def __init__(self, threadID, name, q):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.q = q
#
#     def run(self):
#         print("开启线程：" + self.name)
#         process_data(self.name, self.q)
#         print("退出线程：" + self.name)
#
#
# def process_data(threadName, q):
#     while not exitFlag:
#         queueLock.acquire()
#         if not workQueue.empty():
#             data = q.get()
#             queueLock.release()
#             print("%s processing %s" % (threadName, data))
#         else:
#             queueLock.release()
#         time.sleep(1)
#
#
# threadList = ["Thread-1", "Thread-2", "Thread-3"]
# nameList = ["One", "Two", "Three", "Four", "Five"]
# queueLock = threading.Lock()
# workQueue = queue.Queue(10)
# threads = []
# threadID = 1
#
# # 创建新线程
# for tName in threadList:
#     thread = myThread(threadID, tName, workQueue)
#     thread.start()
#     threads.append(thread)
#     threadID += 1
#
# # 填充队列
# queueLock.acquire()
# for word in nameList:
#     workQueue.put(word)
# queueLock.release()
#
# # 等待队列清空
# while not workQueue.empty():
#     pass
#
# # 通知线程是时候退出
# exitFlag = 1
#
# # 等待所有线程完成
# for t in threads:
#     t.join()
# print("退出主线程")

# ---------------------------------------------------------------------------
# inputData = 0
#
# lock = threading.Lock()
#
#
# def change_it(n):
#     # 先存后取，结果应该为0:
#
#     global inputData
#
#     inputData = inputData + n
#     print(inputData)
#     inputData = inputData - n
#     print(inputData)
#
#
# def run_thread(n):
#     for i in range(10):
#         #获取锁
#         lock.acquire()
#         try:
#             change_it(n)
#             print(inputData)
#         finally:
#             lock.release()
#
#
# t1 = threading.Thread(target=run_thread, args=(1,))
#
# t2 = threading.Thread(target=run_thread, args=(8,))
#
# t1.start()
#
# t2.start()
#
# t1.join()
#
# t2.join()
#
# print(inputData)

# ------------------------------------------------------------------

class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        threadLock.acquire()
        print_time(self.name, self.counter, 3)
        # 释放锁，开启下一个线程
        threadLock.release()


def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        os.system("yolo_video_person.py")
        counter -= 1


threadLock = threading.Lock()
threads = []

# 创建新线程
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# 开启新线程
thread1.start()
thread2.start()

# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)


# 等待所有线程完成
for t in threads:
    t.join()
print("退出主线程")

# -------------------------------------------------------------------------
# 监听模式
# class Observer(object):
#     def update(self, heatHeater):
#         pass
#
#
# class Observable(object):
#     def __init__(self):
#         self.__observers = []
#
#     def addObserver(self, observer):
#         self.__observers.append(observer)
#
#     def notify(self):
#         for o in self.__observers:
#             o.update(self)
#
#
# class heatWater(Observable):
#     def __init__(self):
#         super().__init__()
#         self.__waterHeater = 25
#
#     def getWatterHeater(self):
#         return self.__waterHeater
#
#     def heater(self, waterHeater):
#         print(waterHeater)
#         self.__waterHeater = waterHeater
#         self.notify()
#
#
# class tvmonitor(Observer):
#     def update(self, heatHeater):
#         if (heatHeater.getWatterHeater() >= 1):
#             os.system("yolo_video_tvmonitor.py")
#
#
# class person(Observer):
#     def update(self, heatHeater):
#         if (heatHeater.getWatterHeater() >= 5):
#             os.system("yolo_video_person.py")
#
#
# t = heatWater()
# t.addObserver(tvmonitor())
# t.addObserver(person())
# t.heater(int(input("input1:")))
# t.heater(int(input("input2:")))
