from tkinter import Tk, Frame
# import keyboard
from pynput import keyboard

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


# print(1)
# keyboard.wait('enter')
# print(2)
# keyboard.wait('w')
# print(3)


# def test1():
#     print("Is 1")
#
#
# if __name__ == '__main__':
#     keyboard.add_hotkey('enter', test1)
#     keyboard.wait()  # 阻塞进程


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
    except AttributeError:
        print('special key {0} pressed'.format(key))


def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


# Collect events until released

keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
lst = [keyboard_listener]

for t in lst:
    t.start()

for t in lst:
    t.join()
