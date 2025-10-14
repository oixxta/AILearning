"""
플라스크로 간단한 웹서버 구현
"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Hello Flask Worlds!'

@app.route('/first', methods=['GET'])
def first():
    name = request.args.get("name")
    return f"<h1>hello <span style='color:blue'>{name}</span></h1>"
"""
http://127.0.0.1:5000/first?name=jsk 로 접속할 경우, first 매서드 호출.
파라미터가 웹 페이지 바디의 안으로 들어감.
"""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  #호스트와 포트 파라미터를 받음.

"""
http://localhost:5000/ 혹은 http://127.0.0.1:5000 혹은 http://192.168.0.6:5000 로 페이지 열 수 있음. 
루프백(127.0.0.1:5000, 자기 자신을 호출함.)

서버 역할의 컴퓨터 말고 다른 컴퓨터에서 서비스 중인 서버에 접근하려면 http://192.168.0.6:5000로 접근해야 함.

같은 망 안에서만 가능
"""