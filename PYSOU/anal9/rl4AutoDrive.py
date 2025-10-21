"""
자동차 자율주행

1차원 공간에서 위치를 이동하며 중앙에 머무는 것이 목표임.
환경은 1차선 도로로 가정, 에이전트(차량)는 좌우에 치우치지 않게 중앙 유지.
액션은 총 세 개(왼쪽으로 : 오른쪽에 치우칠 경우, 중앙 유지, 오른쪽으로 : 왼쪽에 치우칠 경우)
"""
import numpy as np
import random
import matplotlib.pyplot as plt

### 환경 잡기
state_space = np.linspace(-1.0, 1.0, 11)    # 상태공간, -1부터 1까지 11등분
print(state_space)  # [-1.  -0.8 -0.6 -0.4 -0.2  0.   0.2  0.4  0.6  0.8  1. ]
action_space = [-1, 0, 1]                   # 행동 공간, 좌, 중앙, 우, 3가지.

q_table = np.zeros((len(state_space), len(action_space))) # 11행 3열 짜리 2차원 배열 생성(모두 0으로 차있는.)
#print(q_table)


### 학습 하이퍼 파라미터 준비
alpha = 0.1     #알파(α)값, 학습률
gammer = 0.9    #감마(γ)값, 할인률
epsilon = 0.1   #엡실론(ε)-greedy값, 초기 탐험률, 탐험률 10%
episodes = 500  # 탐험 횟수

def get_state_index(position):      #연속적인 값을 이산화 해서 인덱스로 반환
    return np.argmin(np.abs(state_space - position))

#print(get_state_index(-0.1))    # 4
#print(get_state_index(0.123))   # 6

def get_reward(position):           #보상함수, 중앙에 가까울수록, 보상이 큼, 멀수록 보상이 작음.
    return -abs(position)   #절댓값이 커봤자 마이너스로 인해 오히려 작은 수가 되어버림.(음수보상)

def step_function(position, action):    #환경의 동작에 대한 정의
    position += action * 0.1            #현재 위치에 행동(action)을 반영함. 행동은 -1, 0, 1 -> 이동량은 -0.1, 0., 1.0
    position = np.clip(position, -1.0, 1.0)     #position 값은 -1.0 <= position <= 1.0 (범위 고정)
    reward = get_reward(position)       #이동 후 위치에 대한 보상 반환.
    return position, reward             #새로운 위치와 그에 대한 보상을 반환함.

reward_list = []    # 리워드들을 모아놓을 기억장치

### Agent의 학습 루프 - 행동을 선택하고 학습함.
for ep in range(episodes):
    position = np.random.uniform(-1.0, 1.0)          #귣등 분포
    total_reward = 0

    for _ in range(50):     # 한 개의 에피소드마다 50번의 action을 반복함.
        state_idx = get_state_index(position)

        if random.random() < epsilon:               #탐험(Exploration)
            action_idx = random.choice([0, 1, 2])
        else:                                       #이용(Exploitation)
            action_idx = np.argmax(q_table[state_idx])

        action = action_space[action_idx]   #선택된 행동을 환경에 적용함.
        next_position, reward = step_function(position, action) #새로운 위치와 그에 대한 보상을 반환.
        next_state_idx = get_state_index(next_position) # 다음 인덱스에 대한 위치 계산

        # Q-table 업데이트 : 벨만 방정식 적용
        best_next_q = np.max(q_table[next_state_idx])
        q_table[state_idx, action_idx] += alpha * (reward + gammer * best_next_q - q_table[state_idx, action_idx])
        
        position = next_position    #현재위치 갱신
        total_reward += reward      #총 보상 누적

    reward_list.append(total_reward)    #에피소드마다 받은 총 보상을 저장(기록)함 : 시각화를 위해.

    if ep % 50 == 0:        #50 에피소드 마다 성능요약 출력
        initial_avg = np.mean(reward_list[:50])
        final_avg = np.mean(reward_list[-50:])
        max_reward = np.max(reward_list)
        min_reward = np.min(reward_list)
        print("성능 요약 : ")
        print(f'-initial 50 episodes avg reward : {initial_avg:.3f}')
        print(f'-final 50 episodes avg reward : {final_avg:.3f}')
        print(f'-max_reward : {max_reward:.3f}')
        print(f'-min_reward : {min_reward:.3f}')

        # 보상 증가여부(성능 개선 여부)
        if final_avg > initial_avg:
            print(f'모델이 개선됨 : (+{final_avg - initial_avg:.3f})')
        else:
            print(f'모델이 개선되지 않음. 파라미터 조정 필요함.')


### 보상 변화 시각화 하기
plt.figure(figsize=(10, 5))
plt.plot(reward_list, label='episode rewards')
plt.axhline(y = 0, color = 'gray', linestyle='--', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('total_reward')
plt.grid(True)
plt.legend()
plt.show()
plt.close()


### 에피소드 50개 단위로 평균 보상 시각화 하기
window = 50
avg_rewards = [] 

for i in range(0, len(reward_list), window):
    chunk = reward_list[i:i + window]
    avg = np.mean(chunk)
    avg_rewards.append(avg)

plt.figure(figsize=(10, 5))
plt.plot(range(0, len(reward_list), window), avg_rewards, marker='o', label='avg_reward(50 ep)')
plt.xlabel('Episode')
plt.ylabel('avg_reward')
plt.grid(True)
plt.legend()
plt.show()
plt.close()
