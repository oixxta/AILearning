from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import math
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

turret_rotate_complete = False
#start_rotate_chassis = False

target_x = 180
target_z = 160
target_radius = 10       #반지름

# 직전 좌표/시간 (속도 추정용)
_last_pos = None         # (x, y, z)
_last_ts  = None         # time.time()
# EMA(지수이동평균)용 저장
_speed_ema = None

# 유니티 유닛→미터 스케일 (보통 1유닛=1m면 1.0)
WORLD_TO_METER = 1.0
# EMA 스무딩 계수(0~1). 높을수록 최신값을 더 반영
SPEED_EMA_ALPHA = 0.8



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

### 초당 속도 계산
def get_my_speed(x: float, y: float, z: float, now: float = None) -> float:
    """
    현재 좌표(x,y,z)와 시각(now)로부터 플레이어 속도(m/s)를 추정해 반환합니다.
    내부적으로 직전 좌표/시간을 기억해 거리/시간으로 계산합니다.
    EMA 스무딩을 적용해 노이즈를 완화합니다.
    """
    global _last_pos, _last_ts, _speed_ema

    if now is None:
        now = time.time()

    # 첫 프레임이면 속도를 계산할 수 없으므로 0으로 초기화하고 상태만 저장
    if _last_pos is None or _last_ts is None:
        _last_pos = (x, y, z)
        _last_ts = now
        if _speed_ema is None:
            _speed_ema = 0.0
        return _speed_ema

    # 이동량/시간 계산
    dx = (x - _last_pos[0]) * WORLD_TO_METER
    dy = (y - _last_pos[1]) * WORLD_TO_METER
    dz = (z - _last_pos[2]) * WORLD_TO_METER
    dt = max(1e-3, now - _last_ts)  # 0으로 나눔 방지

    inst_speed = math.sqrt(dx*dx + dy*dy + dz*dz) / dt  # m/s

    # EMA 스무딩
    if _speed_ema is None:
        _speed_ema = inst_speed
    else:
        _speed_ema = SPEED_EMA_ALPHA * inst_speed + (1 - SPEED_EMA_ALPHA) * _speed_ema

    # 상태 업데이트
    _last_pos = (x, y, z)
    _last_ts = now

    return _speed_ema

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
    
    # 시작하기에 앞서, 포탑을 E방향으로 90도 회전
    if turret_rotate_complete == False:
        if turret_x <= 88:
            command.update({"turretQE": {"command": "E", "weight": 1.0}})
        elif turret_x >= 90:
            command.update({"turretQE": {"command": "Q", "weight": 0.3}})
        else:
            command.update({"turretQE": {"command": "STOP", "weight": 1.0}})
            turret_rotate_complete = True

    else:
        # 무지성으로 직진, 속도는 시속 30km 까지만.
        if speed_kmh < 25:
            command.update({"moveWS": {"command": "W", "weight": 1.0}})
        else:
            command.update({"moveWS": {"command": "W", "weight": 0.5}})


        # 차체 회전 : 목표물과의 거리에 따라 
        distance = calculate_distance(pos_x, pos_z, target_x, target_z)
        if distance > target_radius:
            command.update({"moveAD": {"command": "D", "weight": 0.85}})
        elif distance < target_radius:
            command.update({"moveAD" : {"command" : "A", "weight": 0.85}})
        else:
            command.update({"moveAD": {"command": "STOP", "weight": 0.85}})

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
