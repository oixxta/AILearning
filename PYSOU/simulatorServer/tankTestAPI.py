"""
READ ME : 개발 착수 첫날의 탐색개발 수도코드를 실시했던 시점과 달리, 방배동밸리 대표가 시연한 각종
시연자료 반영 및 스파게티 코드 방지, 더 직관적인 모듈별 역할 분할 및 향후 깃허브 협업을 고려해
모듈과 코드 구조를 탐색개발 시점에서 전면 개편함. 추가로, 역할중복 등으로 필요성이 없어진 정찰모듈은 삭제.

경고 : 위 코드는 더미코드임. 각 모듈의 세부 구현은 개발 착수 시 필요에 따라 전면 개편이 예상됨.


<각 모듈별 구상 및 기대 역할>
 
회피모듈 (evasion_module) (get_action 안의 모든 모듈들 중 최 우선 실행)
 1. 해당 모듈은 다른 모듈이 중단될 경우 무조건적으로 최우선 실행. get_action에서 최우선으로 Pre/Post 두 번 안전망
 2. 라이다 센서(8채널짜리, 차체에 장착) 상 12시 방향이 붉은색일 경우 차체 회전
    (회전 방향은 좌/우 평균 거리 비교해서 재연성이 높은 쪽으로)
 3. 라이다 센서 상 12시 방향 좌우 30도가 녹색일 경우, 직진 가능으로 판단, 회피모듈 종료(아마도 다시 주행모듈로 돌아갈거)

탐지모듈 (detect_module) (상시 실행)
 1. (상시호출. 아마도 전투, 주행 모듈과 함께 실행될 수 있게 조건문으로 처리해야 할듯?) 
    카메라의 YOLO로 각종 오브젝트(장애물, 적, 아군)전차 구분.
 2. 해당 모듈의 목적은 다른 모듈을 실행할 수 있게 하는 글로벌 플래그 선언이 주 목적. 
    화면 상에 구분된 오브젝트에 따라 다른 모듈을 실행시키는 get_action의 조건문을 만족시키게 함.
    상태 플래그: has_target, front_clear, collision_risk, map_ready 등…

전투모듈 (combet_module)
 1. (조건문에 의거해 전투모듈로 이행될 경우 : 교전 가능거리) 적 전차와 내 전차의 거리와 각도 계산
    (양 측의 x, y, z 좌표에 의거)
 2. 위 계산에 의거해, 적을 명중할 수 있게 포탑 및 포신을 회전해서 조준 (PID로 조준)
    2-1 전방에 아군 혹은 장애물 존재 유무 확인?
    2-2 적 전차가 움직이는 상태일경우(적 전차의 속도가 0이 아님), 예측샷을 위한 조준(?)
 3. 사격 (시뮬레이션 상 7초에 한번 씩 사격 가능)
 4. 적 전차의 hp가 0 (혹은 전차가 파괴, 시뮬레이션 상으론 delete) 될 때까지 계속 사격
 5. 적 전차가 파괴될 경우 전투모듈 종료.

주행모듈 (drive_module) (경로 계산)
 1. 현재 탱크 위치와 적 전차 좌표를 기반으로 경로 생성
 2. A* 알고리즘을 활용해 지형상의 최적 경로 산출
 3. 직선구간에선 가속 실시(최대속도까지, 시뮬레이터 상 시속 65km)
 4. 곡선구간에선 곡선 예정 지점 앞에서 부터 감속 후(시속 25km?) 차체 회전
    (조건문 필요? 아마 경로를 기반으로 차체 회전방향 결정)
    4-1 차체 회전 완료 후엔 포탑도 차체에 맞게 회전 -> 필요 없음. 시뮬레이터 상 포탑은 무조건 차체와 같이 동기화되어있음.

 
모듈의 내에서 다른 모듈을 부르는 일은 사라짐. 전부 get_action 내의 조건문 등으로 모듈 호출조건 지정.
"""

from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import math

app = Flask(__name__)
model = YOLO('yolov8n.pt')


"""
각 모듈(탐지, 전투, 주행, 회피)은 장기적으로 다른 파일로 분리, 
실 개발 개시 시, import 방식으로 불러오게 바꿈. (Github 협업 시 merge 충돌 방지를 위해)
"""

################################################################
# 유틸 : 기존 API에서 각 모듈에 필요한 것만 선별해서 따로 메서드 정의  #
################################################################
def clamp_weight(w):
    try:
        w = float(w)
    except:
        return 1.0
    return max(0.0, min(1.0, w))

def ensure_action_schema(a):
    """API 3.2.5 응답 스키마를 항상 만족하도록 보정"""
    return {
        "moveWS":  {"command": a.get("moveWS", {}).get("command", ""),  "weight": clamp_weight(a.get("moveWS", {}).get("weight", 0.0))},
        "moveAD":  {"command": a.get("moveAD", {}).get("command", ""),  "weight": clamp_weight(a.get("moveAD", {}).get("weight", 0.0))},
        "turretQE":{"command": a.get("turretQE", {}).get("command", ""), "weight": clamp_weight(a.get("turretQE", {}).get("weight", 0.0))},
        "turretRF":{"command": a.get("turretRF", {}).get("command", ""), "weight": clamp_weight(a.get("turretRF", {}).get("weight", 0.0))},
        "fire": bool(a.get("fire", False))
    }

def vec_sub(a, b):
    return {"x": a["x"]-b["x"], "y": a["y"]-b["y"], "z": a["z"]-b["z"]}

def vec_norm(v):
    return math.sqrt(v["x"]**2 + v["y"]**2 + v["z"]**2)

def yaw_pitch_to(a, b):
    """ego a -> target b 방향의 yaw(수평), pitch(수직) [deg] 근사 계산"""
    d = vec_sub(b, a)
    yaw = math.degrees(math.atan2(d["x"], d["z"]))                 # 전방 z축 기준
    horiz = math.sqrt(d["x"]**2 + d["z"]**2)
    pitch = math.degrees(math.atan2(d["y"], horiz))
    return yaw, pitch

def wrap_angle_deg(x):
    """[-180, +180]로 래핑"""
    while x > 180: x -= 360
    while x < -180: x += 360
    return x

@app.route('/detect', methods=['POST'])
def detect_endpoint_stub():
    # TODO: YOLO 추론 결과를 실제로 반환하도록 연결
    return jsonify([]), 200



###############################
#   탐지모듈 (요청 스코프 전용)
###############################
def detect_module(ctx):
    print("탐지모듈 호출")
    """
    목적:
      - YOLO detections + LiDAR로 전역 플래그와 표적 후보 생성
      - get_action 조건문에서 사용할 불리언/수치 플래그 제공
    출력 형식:
      {
        "flags": {"has_target":bool,"front_clear":bool,"collision_risk":bool,"map_ready":bool},
        "env": {"min_front": float},
        "targets": [ {"id":int,"class":str,"pos":{x,y,z}|None,"dist":float,"speed":float,"conf":float,"threat":float} ],
        "ogm": None | "<occupancy_grid_repr>"
      }
    """
    dets = ctx.get("detections", []) or []
    lidar = ctx.get("lidar", []) or []
    ego_pos = ctx["ego"]["pos"]

    # 전방(±30°) 최소거리 계산 → front_clear / collision_risk 플래그
    front_distances = [float(p.get("distance", 999.0))
                       for p in lidar
                       if -30.0 <= float(p.get("angle", 0.0)) <= 30.0]
    min_front = min(front_distances) if front_distances else 999.0
    D_SAFE = 8.0       # 전방 주행 안전 임계
    D_EMER = 4.0       # 즉시 회피 임계
    front_clear = (min_front >= D_SAFE)
    collision_risk = (min_front < D_EMER)

    # YOLO 추정에서 적(또는 차량류) 후보 골라 타깃 만들기
    # 명세상 클래스 예시는 자유로움. 필요시 {0:"person",2:"car",7:"truck"} 등 매핑.
    targets = []
    for i, d in enumerate(dets):
        cls_name = d.get("className") or d.get("cls")
        conf = float(d.get("conf", d.get("confidence", 1.0)))
        if conf < 0.35:
            continue
        # 월드 좌표 pos가 없을 수 있음 → TODO: LiDAR/투영 융합으로 pos 근사
        pos = d.get("pos")
        dist = vec_norm(vec_sub(pos, ego_pos)) if pos else 9999.0
        # 위협도 간단 가중 (거리 짧을수록↑, 신뢰도↑)
        threat = max(0.0, min(1.0, (1.0 / (1.0 + dist/100.0)) * 0.6 + conf * 0.4))
        targets.append({
            "id": i,
            "class": cls_name or "unknown",
            "pos": pos,                       # TODO: pos 없으면 추정
            "dist": dist,
            "speed": 0.0,                     # TODO: 상태 없으므로 속도 추정 불가(추후 칼만/트래킹)
            "conf": conf,
            "threat": threat
        })

    # enemyPos가 따로 주어질 경우 fallback 표적
    if not targets and ctx.get("enemy", {}).get("pos"):
        epos = ctx["enemy"]["pos"]
        dist = vec_norm(vec_sub(epos, ego_pos))
        targets.append({
            "id": 999,
            "class": "enemy",
            "pos": epos,
            "dist": dist,
            "speed": ctx.get("enemy", {}).get("speed", 0.0),
            "conf": 1.0,
            "threat": 0.8
        })

    det_out = {
        "flags": {
            "has_target": bool(targets),
            "front_clear": front_clear,
            "collision_risk": collision_risk,
            "map_ready": False  # TODO: Occupancy Grid 생성 시 True
        },
        "env": {"min_front": min_front},
        "targets": sorted(targets, key=lambda t: t["threat"], reverse=True),
        "ogm": None  # TODO: OGM/코스트맵 구축(장애물 지도) → drive에서 A* 사용
    }
    return det_out


###############################
# 전투모듈
###############################
def combat_module(ctx, det_out):
    print("전투모듈 호출")
    """
    출력 형식:
      {
        "in_range": bool,
        "turretQE": {"command":"Q|E|", "weight":float},
        "turretRF": {"command":"R|F|", "weight":float},
        "fire": bool,
        "engage_dist": float,   # 접근을 멈추고 교전 유지하고 싶은 거리
        "keepout_dist": float   # 너무 붙었을 때 이탈하고 싶은 거리
      }
    """
    ego_pos = ctx["ego"]["pos"]
    turret_az = float(ctx["turret"]["az"])
    turret_el = float(ctx["turret"]["el"])
    targets = det_out.get("targets", []) or []

    if not targets:
        return {
            "in_range": False,
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
            "engage_dist": 180.0,  # 기본 정책값 (m)
            "keepout_dist": 40.0
        }

    tgt = targets[0]
    if not tgt.get("pos"):
        # TODO: pos 추정 불가 시 조준 불가 → 접근 유도
        return {
            "in_range": False,
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
            "engage_dist": 180.0,
            "keepout_dist": 40.0
        }

    # 거리/각도 계산
    dist = tgt["dist"]
    yaw_t, pitch_t = yaw_pitch_to(ego_pos, tgt["pos"])
    d_az = wrap_angle_deg(yaw_t - turret_az)
    d_el = wrap_angle_deg(pitch_t - turret_el)

    # 조준 방향(간단 스위치). 실제론 PID → 명령/가중치 맵핑 필요
    qe = "E" if d_az > 0 else ("Q" if d_az < 0 else "")
    rf = "R" if d_el > 0 else ("F" if d_el < 0 else "")
    qe_w = min(1.0, max(0.1, abs(d_az)/30.0)) if qe else 0.0
    rf_w = min(1.0, max(0.1, abs(d_el)/15.0)) if rf else 0.0

    # 유효 교전 거리
    ENGAGE = 180.0
    KEEPOUT = 40.0
    in_range = (dist <= ENGAGE)

    # 사격 게이팅
    AZ_GATE = 1.0
    EL_GATE = 0.5

    # 아군/장애물 라인오브파이어 체크 → TODO: 시뮬레이터 충돌/레이캐스트 필요
    line_of_fire_clear = True  # TODO

    # 7초 쿨다운 → TODO: /get_action은 요청 스코프라 내부 상태가 없어 타이머 유지 불가
    cooldown_ready = True  # TODO: 외부 상태/타임스탬프 관리 필요

    fire_ready = (in_range and abs(d_az) < AZ_GATE and abs(d_el) < EL_GATE
                  and line_of_fire_clear and cooldown_ready)

    return {
        "in_range": in_range,
        "turretQE": {"command": qe, "weight": qe_w},
        "turretRF": {"command": rf, "weight": rf_w},
        "fire": bool(fire_ready),
        "engage_dist": ENGAGE,
        "keepout_dist": KEEPOUT
    }


###############################
# 주행모듈
###############################
def drive_module(ctx, det_out, combat_out):
    print("주행모듈 호출")
    """
    출력 형식:
      {
        "moveWS": {"command":"W|S|STOP|", "weight":float},
        "moveAD": {"command":"A|D|",      "weight":float}
      }
    정책:
      - target가 있고 in_range=False → target으로 접근
      - target 없고 destination 있으면 → destination 향해 이동
      - 그 외 → 순찰/유지 (여기선 정지)
    경로계획:
      - API만으론 지도/OGM 없음 → A*는 TODO. 현재는 헤딩 기반 간이 제어.
    속도:
      - 직선 가속/곡선 감속 → TODO: 곡률 기반; 여기서는 회전량 기반 weight로 근사
    """
    ego_pos = ctx["ego"]["pos"]
    # 1) 목표 결정
    goal = None
    targets = det_out.get("targets", []) or []
    if targets and not combat_out.get("in_range", False) and targets[0].get("pos"):
        goal = targets[0]["pos"]  # 접근
    elif ctx.get("destination"):
        goal = ctx["destination"]
    else:
        # TODO: 정찰(웨이포인트) 통합 시 여기서 goal 제공
        pass

    # 2) 목표 없으면 기본 정지
    if not goal:
        return {"moveWS": {"command": "W", "weight": 0.6},
                "moveAD": {"command": "", "weight": 0.0}}

    # 3) A* 글로벌 경로 → TODO: det_out["ogm"] 필요. 지금은 헤딩 기반으로 근사
    yaw_to_goal, _ = yaw_pitch_to(ego_pos, goal)
    # 차량 바디 yaw가 없음 → TODO: ctx["ego"]["yaw"] 제공 시 정확 제어 가능
    body_yaw = 0.0
    d_yaw = wrap_angle_deg(yaw_to_goal - body_yaw)

    # 4) 회전/전진 간단 규칙
    turn_cmd = "A" if d_yaw < -3.0 else ("D" if d_yaw > 3.0 else "")
    turn_w = min(1.0, max(0.3, abs(d_yaw)/45.0)) if turn_cmd else 0.0

    # 거리로 전/후진/정지 결정
    dist = vec_norm(vec_sub(goal, ego_pos))
    if dist > combat_out.get("engage_dist", 180.0):   # 아직 멈출 거리보다 멀다 → 전진
        move_cmd, move_w = "W", 0.7
    elif dist < combat_out.get("keepout_dist", 40.0): # 너무 가까움 → 후퇴
        move_cmd, move_w = "S", 0.7
    else:
        move_cmd, move_w = "STOP", 1.0

    # 5) 직선/곡선 속도차 → TODO: 곡률/경로 기반 속도 프로파일링
    return {"moveWS": {"command": move_cmd, "weight": move_w},
            "moveAD": {"command": turn_cmd, "weight": turn_w}}


###############################
# 회피모듈 (최우선, 다른 모듈을 실행중에도 언제든 덮어씌울 수 있게 해야 함.)
###############################
def evasion_module(ctx, proposed):
    print("회피모듈 호출")
    """
    - 전방 LiDAR 즉시 위험 시: 상위 명령 덮어쓰기 (후진 + 측면 회전), fire=False
    - "12시 적색/녹색" 개념은 UI 표현이므로 실제 판단은 거리 임계로 구현
    - 좌/우 회피 방향은 랜덤 대신 여유 큰 쪽 권장(재현성↑) → 여기선 좌/우 평균 거리 비교
    """
    lidar = ctx.get("lidar", []) or []

    # 전방(±30°), 좌(30~90°), 우(-90~-30°) 평균거리
    front = [float(p.get("distance", 999)) for p in lidar if -30 <= float(p.get("angle", 0)) <= 30]
    left  = [float(p.get("distance", 999)) for p in lidar if  30 < float(p.get("angle", 0)) <= 90]
    right = [float(p.get("distance", 999)) for p in lidar if -90 <= float(p.get("angle", 0)) < -30]

    min_front = min(front) if front else 999.0
    D_SAFE = 8.0
    if min_front < D_SAFE:
        avg_left  = sum(left)/len(left) if left else 0.0
        avg_right = sum(right)/len(right) if right else 0.0
        # 여유 큰 쪽으로 회전
        turn = "A" if avg_left > avg_right else "D"
        override = {
            "moveWS": {"command": "S", "weight": 0.8},
            "moveAD": {"command": turn, "weight": 0.6},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }
        return ensure_action_schema(override)

    # 전방이 충분히 비었으면 그대로 통과
    return ensure_action_schema(proposed)


###############################
# Get Action
###############################
@app.route('/get_action', methods=['POST'])
def get_action():
    """
    전체 파이프라인:
      1) 요청 JSON 파싱 → ctx(요청 스코프 컨텍스트) 생성
      2) detect → combat/drive 분기
      3) 합성: 주행 + (교전 터렛) + 사격  (정찰은 Navigation에 흡수 가정)
      4) evasion(최우선 안전 레이어)로 최종 수정
      5) 최종 명령 JSON 응답
    """
    data = request.get_json(force=True)

    # 필수 필드(포지션/터렛) 추출 + 디폴트
    position = data.get("position", {}) or {}
    turret = data.get("turret", {}) or {}

    pos_x = float(position.get("x", 0.0))
    pos_y = float(position.get("y", 0.0))
    pos_z = float(position.get("z", 0.0))

    turret_x = float(turret.get("x", 0.0))  # azimuth(가로)
    turret_y = float(turret.get("y", 0.0))  # elevation(세로)

    print(f"▦▦▦ Position received ▦▦▦ : x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"◎◎◎ Turret received ◎◎◎ : x={turret_x}, y={turret_y}")

    # 요청 스코프 컨텍스트
    ctx = {
        "time": float(data.get("time", 0.0)),
        "ego": {
            "pos": {"x": pos_x, "y": pos_y, "z": pos_z},
            "yaw": 0.0,  # TODO: 시뮬레이터가 제공하면 사용
            "speed": float(data.get("playerSpeed", 0.0)),
            "health": float(data.get("playerHealth", 100.0))
        },
        "turret": {"az": turret_x, "el": turret_y},
        "enemy": {
            "pos": data.get("enemyPos"),
            "speed": float(data.get("enemySpeed", 0.0))
        },
        "lidar": data.get("lidarPoints", []),
        "detections": data.get("detections", []),   # /detect 결과를 클라가 실어줄 경우
        "destination": data.get("destination"),
        "obstacles": data.get("obstacles", [])      # 3.2.8 포맷 준수 시 사용 가능
    }

    # 1) 탐지 (상시)
    det_out = detect_module(ctx)

    # 2) 전투/주행 분기
    cmb_out = combat_module(ctx, det_out) if det_out["flags"]["has_target"] else {
        "in_range": False, "turretQE": {"command":"", "weight":0.0},
        "turretRF": {"command":"", "weight":0.0}, "fire": False,
        "engage_dist": 180.0, "keepout_dist": 40.0
    }

    drv_out = drive_module(ctx, det_out, cmb_out)

    # 3) 합성: 전투 터렛 우선, 없으면 빈 유지
    proposed = ensure_action_schema({
        "moveWS": drv_out["moveWS"],
        "moveAD": drv_out["moveAD"],
        "turretQE": cmb_out["turretQE"],
        "turretRF": cmb_out["turretRF"],
        "fire": cmb_out["fire"]
    })

    # 4) 회피(최우선)로 최종 수정/덮어쓰기
    final = evasion_module(ctx, proposed)

    print("♤♤♤ Sent Combined Action: ♤♤♤", final)
    return jsonify(final)


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
        "detactMode": True,
        "logMode": False,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("Initialization config sent via /init:", config)
    return jsonify(config)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)