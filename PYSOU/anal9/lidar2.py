# Simple LiDAR RL Environment (Gymnasium-style)
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# 월드/장애물/라이다 설정 ----------
WORLD_W, WORLD_H = 20.0, 15.0             # 월드 크기 (가로, 세로)
OBSTACLES = [(6.0, 4.0, 0.5), (8.0, 10.0, 1.5), (15.0, 5.0, 1.0)]  # 원형 장애물 리스트
NUM_RAYS   = 20                           # LiDAR 레이 개수
FOV        = np.deg2rad(150)              # 시야각 150도 (도 → 라디안)
MAX_RANGE  = 8.0                          # LiDAR 최대 감지 거리
STEP_MARCH = 0.05                         # 레이 전진 단위 거리

def inside_world(x, y):
    return (0.0 <= x <= WORLD_W) and (0.0 <= y <= WORLD_H)   # 월드 경계 내부 여부

def hit_circle(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2                # 점이 원 내부에 있으면 True

def cast_lidar(x, y, theta, num_rays=NUM_RAYS, fov=FOV, max_range=MAX_RANGE, step=STEP_MARCH):
    start = theta - fov/2     # 시야 왼쪽 끝 각도
    angles = start + np.arange(num_rays) * (fov / max(num_rays - 1, 1))  # 균등 분포 각도 배열
    dists = np.full(num_rays, max_range, dtype=np.float32)    # 초기 거리 = 최대거리

    for i, ang in enumerate(angles):                          # 각 레이에 대해 반복
        dist = 0.0 
        hit = False
        while dist < max_range:                               # 최대거리까지 전진
            px = x + math.cos(ang) * dist                     # 레이 끝점 X좌표
            py = y + math.sin(ang) * dist                     # 레이 끝점 Y좌표
            if not inside_world(px, py): hit = True; break    # 경계 벗어나면 충돌
            for (cx, cy, r) in OBSTACLES:                     # 장애물 충돌 검사
                if hit_circle(px, py, cx, cy, r): hit = True; break
            if hit: break
            dist += step                                      # 충돌 없으면 전진
        dists[i] = min(dist, max_range)                       # 충돌 거리 기록
    return dists, angles                                      # 거리 배열, 각도 배열 반환


# Gymnasium 환경 정의 ----------
class SimpleLidarEnv(gym.Env):
    """
    관측: LiDAR 거리 벡터
    행동: {0: 좌회전, 1: 직진, 2: 우회전}
    보상: 목표 접근 +, 충돌 - , 시간 패널티
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)                # 행동 공간 (3가지)
        self.observation_space = spaces.Box(
            low=0.0, high=MAX_RANGE, shape=(NUM_RAYS,), dtype=np.float32
        )                                                     # 관측공간 = 거리 벡터

        self.v = 0.25                                         # 전진 속도
        self.steer_delta = np.deg2rad(8)                      # 회전 단위 각 (8도)
        self.goal = np.array([18.0, 12.0], dtype=np.float32)  # 목표 좌표
        self.goal_radius = 0.6                                # 목표 판정 반경
        self.max_steps = 400                                  # 최대 스텝 수

        self.fig, self.ax = None, None  # 렌더링용 객체
        self._state = None              # [x, y, θ]
        self._prev_goal_dist = None     # 이전 목표 거리
        self._steps = 0                 # 스텝 카운터

    def _get_obs(self):
        x, y, th = self._state
        obs, _ = cast_lidar(x, y, th)   # LiDAR 거리 관측
        return obs.astype(np.float32)

    def _get_info(self):
        x, y, _ = self._state
        d = np.linalg.norm(np.array([x, y]) - self.goal)      # 목표까지의 거리
        return {"goal_dist": float(d), "steps": self._steps}  # info 딕셔너리 반환

    def _collision(self):
        x, y, _ = self._state
        if not inside_world(x, y): return True                # 경계 밖 → 충돌
        for (cx, cy, r) in OBSTACLES:
            if hit_circle(x, y, cx, cy, r + 0.25): return True  # 본체 반경 포함 검사
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = np.array([2.0, 2.0, np.deg2rad(30.0)], dtype=np.float32)  # 초기 상태
        self._steps = 0
        self._prev_goal_dist = np.linalg.norm(self._state[:2] - self.goal)      # 초기 목표거리
        obs = self._get_obs(); info = self._get_info()
        return obs, info

    def step(self, action):
        self._steps += 1
        x, y, th = self._state

        if action == 0: th += self.steer_delta               # 좌회전
        elif action == 2: th -= self.steer_delta             # 우회전
        x += math.cos(th) * self.v                           # 전진 (x 방향)
        y += math.sin(th) * self.v                           # 전진 (y 방향)
        self._state = np.array([x, y, th], dtype=np.float32) # 상태 갱신

        goal_dist = np.linalg.norm(self._state[:2] - self.goal)     # 목표 거리
        progress = self._prev_goal_dist - goal_dist                 # 접근 변화량
        self._prev_goal_dist = goal_dist

        reward = 1.0 * progress - 0.01          # 보상 계산
        terminated, truncated = False, False

        if goal_dist < self.goal_radius:        # 목표 도달
            reward += 1.0; terminated = True
        if self._collision():                   # 충돌 발생
            reward -= 1.0; terminated = True
        if self._steps >= self.max_steps: truncated = True    # 스텝 초과 종료

        obs = self._get_obs(); info = self._get_info()
        return obs, reward, terminated, truncated, info

    # 렌더링 ----------
    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7.5, 5.5))    # 최초 1회만 생성
        ax = self.ax; ax.clear()
        ax.set_xlim(0, WORLD_W); ax.set_ylim(0, WORLD_H)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Simple LiDAR RL Env")

        ax.plot([0, WORLD_W, WORLD_W, 0, 0], [0, 0, WORLD_H, WORLD_H, 0], lw=2) # 경계 사각형

        for (cx, cy, r) in OBSTACLES:                               
            circ = plt.Circle((cx, cy), r, edgecolor='tab:red', facecolor='none', lw=2) # 장애물 원
            ax.add_patch(circ)

        goal = plt.Circle(tuple(self.goal), self.goal_radius,
                          edgecolor='tab:green', facecolor='none', lw=2)
        ax.add_patch(goal)  # 목표 지점

        x, y, th = self._state; L = 0.6
        tri = np.array([      # 에이전트 삼각형
            [x + np.cos(th) * L, y + np.sin(th) * L],
            [x + np.cos(th + 2.5) * L / 1.5, y + np.sin(th + 2.5) * L / 1.5],
            [x + np.cos(th - 2.5) * L / 1.5, y + np.sin(th-2.5) * L / 1.5],
        ])
        ax.fill(tri[:, 0], tri[:, 1], alpha=0.85, color='tab:blue', label='agent')

        obs, angs = cast_lidar(x, y, th)  # LiDAR 빔 시각화
        for d, a in zip(obs, angs):
            ax.plot([x, x + np.cos(a) * d], [y, y + np.sin(a) * d], lw=1, alpha=0.8)

        ax.legend(loc='upper right')
        plt.pause(0.001)                  # 프레임 업데이트

    def close(self):
        if self.fig is not None:
            plt.close(self.fig); self.fig, self.ax = None, None     # 렌더링 자원 해제


if __name__ == "__main__":
    env = SimpleLidarEnv()   
    obs, info = env.reset()  # 초기화
    total_reward = 0.0

    for t in range(500):     # 최대 500 스텝 반복
        action = env.action_space.sample()                          # 랜덤 행동 선택
        obs, reward, terminated, truncated, info = env.step(action) # 환경 단계 진행
        total_reward += reward
        env.render()

        if terminated or truncated:  # 에피소드 종료 조건
            print(f"Episode end at step={t}, total_reward={total_reward:.3f}, info={info}")
            obs, info = env.reset()  # 재시작
            total_reward = 0.0

    env.close()
