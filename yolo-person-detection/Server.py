# -*-coding:utf-8-*-
from flask import Flask
from flask import request
import os

app = Flask(__name__)
# basedir = os.path.abspath(os.path.dirname(__file__))
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


@app.route('/')
def test():
    return '服务器正常运行'


@app.route('/register', methods=['POST'])
def test():
    value = request.form['submit']
    print(value)
    return 'OK'


if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0')
