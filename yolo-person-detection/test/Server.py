# -*-coding:utf-8-*-
from flask import Flask
from flask import request
import os

app = Flask(__name__)


# basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


# @app.route('/')
# def test():
#     return '服务器正常运行'
#
#
# @app.route('/register', methods=['POST'])
# def test():
#     value = request.form['submit']
#     print(value)
#     return 'OK'
#
#
# if __name__ == '__main__':
#     app.run()
    # app.run(host='0.0.0.0')

# 导入socket模块
import socket

# 开启ip和端口
ip_port = ('192.168.1.104', 3295)
# 生成句柄
web = socket.socket()
# 绑定端口
web.bind(ip_port)
# 最多连接数
web.listen(5)
# 等待信息
print('waiting...')
# 开启死循环
while True:
    # 阻塞
    conn, addr = web.accept()
    # 获取客户端请求数据
    data = conn.recv(1024)
    # 打印接受数据 注：当浏览器访问的时候，接受的数据的浏览器的信息等。
    print(data)
    # 向对方发送数据
    conn.send(bytes('<h1>welcome nginx</h1>', 'utf8'))
    # 关闭链接
    conn.close()
