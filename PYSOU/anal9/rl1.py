"""
강화학습 (reinforcement learning)

기초적인 Q-Learning 연습
완전한 환경 모델 없이 model-free 방식으로 벨만 방정식 기반(Q값 갱신)의 근사학습을 사용.

1차원 선에서 좌/우 이동을 하며, 보상 받기 구현
state : 5가지, 이동 : 좌 & 우
"""

import numpy as np
import random

### 공간 정의
state_space = [0, 1, 2, 3, 4]  # 5개의 상태 공간
action_space = [-1, 1]         # 2개의 행동 공간


### Q-tabel 초기화 (상태 : 5, 행동 : 2)
Q = np.zeros([len(state_space), len(action_space)])   # 5 * 2짜리 리스트에 0으로 채움
print(Q)


### 하이퍼 파라미터
alpha = 0.1             #알파(α)값, 학습률
gamma = 0.9             #감마(γ)값, 할인률
epsilon = 1.0          #엡실론(ε)-greedy값, 초기 탐험률, 100% 탐험
epsilon_min = 0.1
epsilon_decay = 0.99
episodes = 100          #탐험 횟수


### 보상함수 정의
def get_reward(state):
    return 10 if state == 4 else 0      #만약, state 값이 4일경우, 10을 보상받음, 아니면 0


### 학습 루프
# 각 에피소드 마다 Q 테이블 갱신하면서 목표 상태(4)에 도달하기 위한 최적의 행동을 취함.
for episode in range(episodes): # 100번 동안 학습 반복
    state = 0
    for stap in range(20):                  #한 개의 에피소드 마다 최대 20번 이동
        # 행동 선택은 ε-greedy를 따름 : (학습 초반에는 다양한 행동, 후반에는 학습된 정책에 따른 최적의 행동 결정)
        if random.random() < epsilon:               #탐험(Exploration)
            action_index = random.randint(0, 1)     #인덱스 값은 0 또는 1, 0이면 왼쪽, 1이면 오른쪽 이동.
        else :                                      #이용(Exploitation)
            action_index = np.argmax(Q[state])      #Q테이블의 값을 그대로 사용해 탐욕적 행동 실시.
        
        action = action_space[action_index]         #탐험 혹은 이용에서 얻은 인덱스 값으로 이동
        next_state = state + action
        #print('next_state : ', next_state)         #실제로 움직이는 것을 확인
        if next_state < 0 or next_state > 4:        #유효범위 바깥(0 미만, 4초과)으로 나가지 못하게 설정
            next_state = state    
        reward = get_reward(next_state)             #state 값이 4일 경우 보상 지급.

        # 벨만 방정식을 적용해 Q-value를 갱신
        old_q = Q[state][action_index]              #현재 추정된 Q값
        next_max = np.max(Q[next_state])            #다음 상태에서 가능한 모든 행동 중 가장 큰 Q값을 선택함.
        Q[state][action_index] = old_q + alpha * (reward + gamma * next_max - old_q)    #벨만방정식 약식 적용

        state = next_state
        if reward == 10:    #목표 상태 도달 시 반복문 중단
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

### 결과출력
print('학습된 Q-table : ')
for s in range(len(state_space)):
    print(f'State {s} : 왼쪽 = {Q[s][0]:.2f}, 오른쪽 = {Q[s][1]:.2f}')

"""
학습된 Q-table :
State 0 : 왼쪽 = 6.34, 오른쪽 = 7.26
State 1 : 왼쪽 = 6.04, 오른쪽 = 8.09
State 2 : 왼쪽 = 6.91, 오른쪽 = 9.00
State 3 : 왼쪽 = 7.46, 오른쪽 = 10.00
State 4 : 왼쪽 = 0.00, 오른쪽 = 0.00
"""
# Q 값이 보상의 전파가 잘 이루어져서 전반적으로 오른쪽으로 향하는 경향을 보임.





