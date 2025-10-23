from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import math
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

target_x = 150
target_y = 0        #실질적으로 미사용.
target_z = 150
target_radius = 25

### 특정 좌표와의 거리 계산
def calculate_distance(x1, z1, x2, z2):
    distance = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    return distance

### 목표 좌표와의 각도 계산
def calculate_angle(x1, z1, x2, z2):
    dx = x2 - x1
    dz = z2 - z1
    angle_rad = math.atan2(dz, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360  # 0~360도로 보정

    return angle_deg


@app.route('/get_action', methods=['POST'])
def get_action():
    global turret_rotate_complete
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})
        
    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    
    
    print("각도 : ", calculate_angle(pos_x, pos_z, target_x, target_z))
    print("거리 : ", calculate_distance(pos_x, pos_z, target_x, target_z))

    command = {  }

    return jsonify(command)


#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detactMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
