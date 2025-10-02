"""
Cartpole(카트폴) 버티기 문제

Q-Learning (off-policy TD Control)방식
MDP(Markov Decision Progress) 기반의 강화학습 알고리즘 사용.
MDP의 5가지 구성 요소
상태(S, state), 환경의 상태
행동(A, action), 에이전트가 행동할 수 잇는 행동
보상(R, reward), 상태-행동에 따른 보상
정책(π, policy), 어떤 상태에서 어떤 행동을 할 지를 결정
상태 전이확률(P, Transition Probability), 상태 전이 확률


카트 위치, 카트 속도, 폴 각도, 폴 각 속도 총 4개의 요소 고려.

한 개의 에피소드 마다 200번의 스탭.
각 진행 스탭마다 막대가 쓰러지지 않을 경우 리워드 1 부여. 따라서, 에피소드당 최대 200리워드 지급.
보상 그래프 시각화

카트폴 시각화에 pip install gymnasium[classic-control] 필요
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML

### 환경 설정
env = gym.make('CartPole-v1')    #카트에 막대(Pole)을 수직으로 세운 채 좌우로 움직여 균형을 유지하는 환경을 제공
print(env.observation_space)     #환경의 관측값 범위(기본값)

### 카트 위치, 카트 속도, 막대 각도, 막대 각도 속도 이산화
obs_space_low = np.array([-2.4, -3.0, -0.5, -2.0])  #카트 위치, 카트 속도, 막대 각도, 막대 각도 속도 최저값
obs_space_high = np.array([2.4, 3.0, 0.5, 2.0])     #카트 위치, 카트 속도, 막대 각도, 막대 각도 속도 최고값

### 상태 공간 이산화 수준 설정. 
#Q-table은 연속적인 상태를 다룰 수 없음. 따라서 구간으로 나눌 수밖에 없음.
state_bins = [6, 12, 6, 12]
q_table = np.zeros(state_bins + [env.action_space.n])
#print(q_table, q_table.shape)       # q_table.shape : (6, 12, 6, 12, 2)

### 상태 이산화(셀 수 있는 값으로) 처리 함수
def discretize_state(state):
    ratios = (state - obs_space_low) / (obs_space_high - obs_space_low)    #정규화
    print('ratios : ', ratios)
    discrete = (ratios * state_bins).astype(int)    # 구간이 선택됨.
    print('discrete : ', discrete)
    return tuple(np.clip(discrete, 0, np.array(state_bins) - 1))    #원본 배율 혀용 최소값, 허용 최댓값을 반환

### discretize_state 함수 결과 테스트
#ex_state = np.array([1.0, 0.5, 0.1, -1.0])
#dis_index = discretize_state(ex_state)
#print('Q-table index : ', dis_index)        #Q-table index :  (4, 7, 3, 3)

### Q-learning의 하이퍼 파라미터(알파값, 감마값) 설정하기
