from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import math
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

### 목표 타겟과의 거리 계산
def calculate_distance(x1, z1, x2, z2):
    distance = math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    print("distance to target : ", distance)
    return distance

### 목표 타겟과의 각도 계산
def calculate_angle(x1, z1, x2, z2):
    dx = x2 - x1
    dz = z2 - z1
    angle_rad = math.atan2(dz, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg < 0:
        angle_deg += 360  # 0~360도로 보정

    return angle_deg

### 차체의 초당 속도 및 각도 계산


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

    speed_mps = get_my_speed(pos_x, pos_y, pos_z, time.time())
    speed_kmh = speed_mps * 3.6
    #print(f"speed = {speed_mps:.3f} m/s ({speed_kmh:.1f} km/h)")

    command = {  }
    

    return jsonify(command)


#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",  # Options: "start" or "pause"
        "blStartX": target_x - target_radius,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": target_z,
        "rdStartX": target_x, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": target_z,
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
