import numpy as np 
import matplotlib.pyplot as plt 
 
# ---------- 환경/에이전트 설정 ---------- 
# 시뮬레이션 세계 즉, LiDAR 센서가 탐지하는 2D 환경의 공간 경계(지도 크기)를 설정. 
WORLD_W, WORLD_H = 20.0, 15.0    # 세계 크기 (가로 20, 세로 15) 
WALL_THICK = 0.5  # 벽 두께(단순히 경계 밖이면 충돌로 처리). 경계판정이나 시각화 개선 시 활용 가능한 변수 
 
# 원형 장애물 (중심 x, 중심 y, 반지름 r)  
OBSTACLES = [ 
    (6.0, 4.0, 1.0),    # (6,4) 위치 반지름 1.0 장애물 
    (12.0, 10.0, 1.5),  # (12,10) 위치 반지름 1.5 장애물 
    (15.0, 5.0, 1.0),   # (15,5) 위치 반지름 1.0 장애물 
] 
 
# 에이전트 초기 상태 (x, y, 바라보는 각도 rad 단위) 
# LiDAR 시뮬레이션의 에이전트(센서가 달린 로봇)의 초기 위치와 방향(heading) 을 설정하는 부분 
agent = dict(x=3.0, y=3.0, theta=np.deg2rad(30))  # 30° 방향을 바라봄 
 
# LiDAR 파라미터 
NUM_RAYS  = 15          # 레이(광선) 개수 
FOV = np.deg2rad(100)     # 시야각 180도 (도(degree)”를 라디안(radian) 으로 변환) 
    # 삼각함수 등에서 각도를 다룰 때, radian은 수학에서 표준 단위, degree는 사람이 직관적으로 쓰는 단위. 
    # 이를 변환하려면 관계식을 사용 :  radians = degrees × (  / 180) 
MAX_RANGE = 5.0    # 탐지 최대 거리 
STEP = 0.05           # 레이 전진 단위 거리 (세밀할수록 정확 ↑). 레이가 0.05m씩 전진하며 충돌 여부 검사 
# 한 레이가 최대 거리 10.0까지 나아가려면 10.0 / 0.05 = 200번의 충돌 체크를 수행
def inside_world(x, y): 
    # (x,y)가 세계 경계 안에 있는지 검사 
    # (x, y) 좌표가 0 ≤ x ≤ 20, 0 ≤ y ≤ 15 범위 안에 있으면 벽안쪽  그 밖이면 충돌(벽에 부딪힘)으로 처리. 
    return (0.0 <= x <= WORLD_W) and (0.0 <= y <= WORLD_H) 
 
def hit_circle(px, py, cx, cy, r): 
    # 점(px,py)이 원(cx,cy,r) 안에 있는지 검사 → 충돌 여부 
    # 점(px, py)이 원 중심(cx, cy) 반지름 r인 장애물 내부 또는 테두리 위에 있으면 True (충돌), 
    # 그렇지 않으면 False (비충돌) 를 반환하는 함수 
    return (px - cx)**2 + (py - cy)**2 <= r**2 
 
# 에이전트 (x,y,θ)에서 시야각 FOV로 NUM_RAYS개의 광선을 쏴서, 각 레이가 처음 부딪히는 지점까지의 거리를 구함. 
# 입력: 시작점(x,y), 바라보는 각 theta, 레이 개수 num_rays, 시야각 fov, 최대거리 max_range, step. 
# 출력: dists — 각 레이의 충돌 거리(충돌 없으면 max_range),  angles — 각 레이의 절대 각도(rad) 
def cast_lidar(x, y, theta, num_rays=NUM_RAYS, fov=FOV, max_range=MAX_RANGE, step=STEP): 
    # (x,y,theta) 위치에서 num_rays개의 레이를 쏘고, 각 레이의 충돌까지 거리를 반환. 
    # 충돌이 없으면 max_range 반환 
    start = theta - fov/2         # 첫 번째 레이 시작 각도 (시야의 왼쪽 끝 각도) 
     
    # angles는 start → start+fov까지 균등 분배된 절대각. max(num_rays-1,1)로 1개일 때도 안전. 
    # 양 끝 포함 분포(좌/우 끝 모두 레이 존재). 
    angles = start + np.arange(num_rays) * (fov / max(num_rays - 1, 1))  # 균등 분배된 레이 각도 배열 
 
    dists = np.full(num_rays, max_range, dtype=float)  # 거리 배열 초기화: 전부 max_range 
 
    for i, ang in enumerate(angles):          # 각 레이에 대해 
        dist = 0.0 
        hit  = False 
        while dist < max_range:             # 최대 거리까지 전진 
            px = x + np.cos(ang) * dist      # 레이 끝점 X 좌표 
            py = y + np.sin(ang) * dist      # 레이 끝점 Y 좌표 
            # 현재 레이 각도 ang로, dist를 step만큼 증가시키며 끝점 (px,py) 이동. 
 
            if not inside_world(px, py):     # 벽 충돌 체크. 월드 밖 → 벽에 부딪힘 
                hit = True    # 경계 밖이면 벽에 충돌로 처리하고 종료. 
                break 
 
            for (cx, cy, r) in OBSTACLES:    # 모든 원형 장애물 검사 
                if hit_circle(px, py, cx, cy, r): 
                    hit = True     # 하나라도 원 내부에 들어가면 충돌. 
                    break
                if hit:        # 충돌이면 종료 
                    break 
            dist += step   # 충돌 없으면 한 걸음 전진 
 
        dists[i] = min(dist, max_range)     # 충돌 거리 기록 (없으면 max_range) 
 
    return dists, angles   # 레이별 거리와 각도(절대각) 반환. 
 
# 2D LiDAR(라이다) 환경을 시각화 
# 입력값 - agent: 로봇(에이전트)의 상태 (x, y, theta 포함된 dict) 
#       - rays_endpoints: 각 ray의 시작점과 끝점 좌표 리스트 → (시각화용) 
def plot_world(agent, rays_endpoints=None): 
    fig, ax = plt.subplots(figsize=(7.5, 5.5)) 
    ax.set_xlim(0, WORLD_W)   # X축 범위 
    ax.set_ylim(0, WORLD_H)   # Y축 범위 
    ax.set_aspect('equal', adjustable='box')  # 가로 세로 비율을 1:1로 유지 (왜곡 방지) 
    ax.set_title("Simple 2D LiDAR") 
 
    # 월드 경계 (벽) 그리기 - 사각형 형태의 월드 경계 박스(벽) 
    ax.plot([0, WORLD_W, WORLD_W, 0, 0], [0, 0, WORLD_H, WORLD_H, 0], lw=2) 
 
    # 장애물(원형) 
    for (cx, cy, r) in OBSTACLES: 
        circ = plt.Circle((cx, cy), r, edgecolor='tab:red', facecolor='none', lw=2) 
        ax.add_patch(circ)   # 축에 원 패치를 추가 - 라이다 감지 대상 장애물(빨간색 원) 
 
    # 에이전트 삼각형(방향 표시) 
    # 에이전트(즉, LiDAR 센서가 달린 로봇 본체)를 삼각형 형태로 그려주는 코드 
    x, y, th = agent["x"], agent["y"], agent["theta"] 
    L = 0.6    # 삼각형의 길이 스케일(scale) 
    tri = np.array([   # 삼각형 좌표 계산 
        [ x + np.cos(th) * L, y + np.sin(th) * L        ],  # 앞쪽. 현재 방향(th)을 따라 L 거리 앞쪽 위치 
        [ x + np.cos(th + 2.5) * L / 1.5, y + np.sin(th + 2.5) * L / 1.5],   
        # 왼쪽 뒤. 본체 중심에서 방향을 왼쪽으로 2.5rad (~143°) 돌린 방향 
        [ x + np.cos(th - 2.5) * L / 1.5, y + np.sin(th - 2.5) * L / 1.5],   
        # 오른쪽 뒤. 본체 중심에서 오른쪽으로 2.5rad (~143°) 돌린 방향 
    ]) 
    ax.fill(tri[:, 0], tri[:, 1], alpha=0.8, color='tab:blue', label='agent') 
 
    # 레이 시각화 : plot_world() 함수에서 LiDAR 센서가 쏜 광선(ray)들을 시각적으로 표시. 
    # 즉, 앞서 계산된 레이의 시작점(에이전트 위치)과 끝점(충돌 지점)을 선으로 그려주는 역할
    if rays_endpoints is not None:  # rays_endpoints는 각 레이의 시작점, 끝점 좌표를 담은 리스트 
        for (x0, y0, x1, y1) in rays_endpoints: 
            # 두 점을 연결하는 선으로 표시 
            ax.plot([x0, x1], [y0, y1], lw=1, alpha=0.8) 
 
    ax.legend(loc='upper right') 
    plt.tight_layout() 
    plt.show() 
 
if __name__ == "__main__": 
    # obs : 각 레이(ray)가 감지한 충돌 거리 배열 (길이 = NUM_RAYS)   obs = [3.2, 4.7, 6.5, ...]   # 거리값 
    # angs : 각 레이의 절대 각도(rad) 배열    angs = [0.0, 0.2, 0.4, ...]  # 각도값 
    obs, angs = cast_lidar(agent["x"], agent["y"], agent["theta"])   # LiDAR 센서 스캔 
    endpoints = []            # 시각화용 레이 끝점 좌표 
    for d, a in zip(obs, angs):   # 각 레이에 대해 
        # 각 레이별로 시작점(x0, y0) 과 충돌 지점(x1, y1) 좌표를 계산 
        x0, y0 = agent["x"], agent["y"]              # 에이전트 위치. 레이 시작점 (에이전트 위치) 
        x1, y1 = x0 + np.cos(a)*d, y0 + np.sin(a)*d  # 레이 끝점. 충돌 지점 좌표 
        # np.cos(a)와 np.sin(a)로 “각도 → 방향 벡터” 변환 후, 거리 d만큼 곱해 끝점을 구함. 
        # 결과적으로 endpoints는 이런 리스트가 된다. 
            # [ 
            #   (3.0, 3.0, 8.0, 3.0),   # 0° 방향으로 5m 
            #   (3.0, 3.0, 6.5, 4.2),   # 30° 방향으로 4.8m 
            #   ... 
            # ]  plot_world()에서 각 선(ray)을 그리는 데 쓰임 
        endpoints.append((x0, y0, x1, y1))  
 
    print("LiDAR observation (distances):")    
    print(np.round(obs, 2))         # 거리값 출력 
        # [ 3.5   3.7   4.05  4.45  5.05  5.85  7.1   9.1  10.   10.   10.    2.5 
        # 2.25  2.2   2.2   2.25  2.45  9.95 10.   10.   10.   10.   10.   10.  
        # 10.   10.   10.   10.   10.    9.55  7.35  6.05] 
    plot_world(agent, endpoints)    # 월드 + 레이 시각화