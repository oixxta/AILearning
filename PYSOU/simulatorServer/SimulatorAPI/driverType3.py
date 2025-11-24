from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import csv
from datetime import datetime
import math

# --- 연결 리스트 형태의 웨이포인트(목표) 관리 ---
class WaypointNode:
    def __init__(self, x, z, arrived=False):
        self.x = float(x)
        self.z = float(z)
        self.arrived = bool(arrived)
        self.next = None

class WaypointList:
    def __init__(self):
        self.head = None
        self.tail = None
        self._len = 0

    def append(self, x, z, arrived=False):
        node = WaypointNode(x, z, arrived)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._len += 1
        return node

    def peek(self):
        return self.head

    def pop(self):
        if not self.head:
            return None
        node = self.head
        self.head = node.next
        if not self.head:
            self.tail = None
        node.next = None
        self._len -= 1
        return node

    def mark_head_arrived(self):
        if self.head:
            self.head.arrived = True
            return True
        return False

    def is_empty(self):
        return self.head is None

    def to_list(self):
        out = []
        cur = self.head
        while cur:
            out.append({'x': cur.x, 'z': cur.z, 'arrived': cur.arrived})
            cur = cur.next
        return out

# 전역 웨이포인트 리스트 인스턴스 (기본값: 빈 리스트)
waypoints = WaypointList()
waypoints.append(182, 84)
waypoints.append(100, 187)
waypoints.append(25, 280)

time = None
distance = None

# Player Tank
player_x = 0
player_y = 0
player_z = 0

player_speed = 0
player_health = 0
player_turret_x = 0
player_turret_y = 0
player_body_x = 0
player_body_y = 0
player_body_z = 0

# Enemy Tank
enemy_x = 0
enemy_y = 0
enemy_z = 0

enemy_speed = 0
enemy_health = 0
enemy_turret_x = 0
enemy_turret_y = 0
enemy_body_x = 0
enemy_body_y = 0
enemy_body_z = 0

app = Flask(__name__)
model = YOLO('yolov8n.pt')

def stabilizer(ally_pos, enemy_pos):
    """
    ally_pos: {'x': , 'y': , 'z': } 아군 전차 위치
    enemy_pos: {'x': , 'y': , 'z': } 적 전차 위치

    반환: {'yaw_absolute', 'pitch_absolute', 'yaw_relative', 'pitch_relative'}
    """
    dx = enemy_pos['x'] - ally_pos['x']
    dz = enemy_pos['z'] - ally_pos['z']
    dy = enemy_pos['y'] - ally_pos['y']

    distance_xz = math.hypot(dx, dz)

    yaw_rad = math.atan2(dx, dz)
    target_yaw = math.degrees(yaw_rad)
    if target_yaw < 0:
        target_yaw += 360

    target_pitch = math.degrees(math.atan2(dy, distance_xz))

    return {
        'yaw': target_yaw,
        'pitch': target_pitch
    }

@app.route('/info', methods=['POST'])
def info():
    global time, distance
    global player_x, player_y, player_z
    global player_speed, player_health, player_turret_x, player_turret_y
    global player_body_x, player_body_y, player_body_z
    global enemy_x, enemy_y, enemy_z
    global enemy_speed, enemy_health, enemy_turret_x, enemy_turret_y
    global enemy_body_x, enemy_body_y, enemy_body_z

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    time = data["time"]
    distance = data["distance"]

    player_x = data["playerPos"]["x"]
    player_y = data["playerPos"]["y"]
    player_z = data["playerPos"]["z"]

    player_speed = data["playerSpeed"]
    player_health = data["playerHealth"]
    player_turret_x = data["playerTurretX"]
    player_turret_y = data["playerTurretY"]
    player_body_x = data["playerBodyX"]
    player_body_y = data["playerBodyY"]
    player_body_z = data["playerBodyZ"]

    enemy_x = data["enemyPos"]["x"]
    enemy_y = data["enemyPos"]["y"]
    enemy_z = data["enemyPos"]["z"]

    enemy_speed = data["enemySpeed"]
    enemy_health = data["enemyHealth"]
    enemy_turret_x = data["enemyTurretX"]
    enemy_turret_y = data["enemyTurretY"]
    enemy_body_x = data["enemyBodyX"]
    enemy_body_y = data["enemyBodyY"]
    enemy_body_z = data["enemyBodyZ"]

    return jsonify({"status": "success", "control": ""})

@app.route('/get_action', methods=['POST'])
def get_action():
    global player_x, player_y, player_z, player_turret_x, player_turret_y
    global enemy_x, enemy_y, enemy_z

    # --- stabilizer로 적 전차를 바라보게 터렛 회전만 수행 ---
    ally_pos = {'x': player_x, 'y': player_y, 'z': player_z}
    enemy_pos = {'x': enemy_x, 'y': enemy_y, 'z': enemy_z}
    result = stabilizer(ally_pos, enemy_pos)
    print("result", result)
    
    # 터렛 회전 명령 계산
    turret_qe_command = ""
    turret_qe_weight = 0.0
    yaw_angle_diff = result['yaw'] - player_turret_x
    if abs(yaw_angle_diff) > 1.0:
        turret_qe_command = "Q" if yaw_angle_diff < 0 else "E"
        turret_qe_weight = 1.0 if abs(yaw_angle_diff) >= 10.0 else 0.1

    turret_rf_command = ""
    turret_rf_weight = 0.0
    pitch_angle_diff = result['pitch'] - player_turret_y
    if abs(pitch_angle_diff) > 1.0:
        turret_rf_command = "R" if pitch_angle_diff > 0 else "F"
        turret_rf_weight = 1.0 if abs(pitch_angle_diff) >= 10.0 else 0.1

    # 주행 명령 없이 터렛 회전만 반환
    action = {
        "moveWS": {"command": "S", "weight": 0.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": turret_qe_command, "weight": turret_qe_weight},
        "turretRF": {"command": turret_rf_command, "weight": turret_rf_weight},
        "fire": False
    }
    return jsonify(action)

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",
        "blStartX": 250,
        "blStartY": 12,
        "blStartZ": 161,
        "rdStartX": 221,
        "rdStartY": 5,
        "rdStartZ": 132,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
