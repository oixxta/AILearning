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
alpha = 0.1
gammer = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.05
episodes = 1500
reward_list = []    # 각 에피소드에서 받은 총 보상
trajectories = []   # 궤적 위치 저장 - 에니메이션 시각화용
best_reward = 0     # 최고의 총 보상 저장

### 
for ep in range(episodes):
    obs, _, = env.reset()   #새로운 에피소드가 시작됨. 목표가 달성되거나 막대가 쓰러지면 종료됨.
    #print(obs)              #[ 0.03485017 -0.00071027  0.00305485  0.00380944] 카트 위치, 카트 속도, 막대 각도, 막대각 속도

    state = discretize_state(obs)   #obs의 값을 상태 이산화(셀 수 있는 값으로) 처리.
    #print(state)
    """
    ratios :  [0.50624557 0.49823922 0.49708212 0.49690533]         #카트 위치, 카트 속도, 막대 각도, 막대각 속도
    discrete :  [3 5 2 5]                                           #위 값을 이산화
    (3, 5, 2, 5)
    """
    total_reward = 0    # 하나의 에피소드에서 받는 보상 누적값 초기화
    trajectory = []     # obs를 매 순간 저장함. (시각화를 위해서)
    for step in range(200):     #하나의 에피소드는 최대 200 스탭까지 실행.
        # 행동을 선택(epsilon greeding, 탐욕적 행동 실시) -> 환경에 행동 전달 -> 새로운 상태, 보상, 종료 여부 반환 -> Q-table 갱신
        # 행동(탐험과 이용) 구현:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()      # 무작위 행동을 선택. (탐험, Exploration)
        else:
            # value state function을 사용
            # 현재 가치에서 가장 높은 Q값을 가지는 행동을 선택함(a = argmax(Q(s, a)))
            action = np.argmax(q_table[state])      # 지금까지 얻은 정보 중 가장 좋은 행동을 선택. 이용(Exploitation)
        #print(action)       # 0 아니면 1의 행동 선택
        #print(env.step(action))    # 5개의 값 반환, (array([-0.43150678,  0.0878814 , -6.882771  , -5.864825  ], dtype=float32), 0.0, True, False, {})
        next_obs, reward, terminated, truncated, info = env.step(action)   # open AI가 제공하는 카트폴의 환경에 action을 적용함. 리턴값 
        """
        next_obs : 카트 위치, 카트 속도, 막대 각도, 막대각 속도
        reward : 보상값
        terminated : 에피소드 종료 조건 충족 여부
        truncated : 시간제한 초과 여부
        """
        done = terminated or truncated  # 막대가 쓰러지든, 시간제한이 끝나든, 둘 중 하나라도 참일 경우, done은 1 (True).
        next_state = discretize_state(next_obs)
        best_next_q = np.max(q_table[next_state])   # 미래 가치 계산
        # 탐험 혹은 이용 -> 행동을 선택, 수행 -> 다음상태 / 보상 / 종료 정보 획득함 ->
        # 상태 이산화 -> 다음 상태에서 가능한 최대 Q값 계산하기(미래 가치)
        #print(best_next_q)      #다음 상태 최적의 Q값

        # Q-table 갱신하기 : 현재 상태에서 어떤 행동을 한 후, 즉시 보상 또는 다음 상태의 기대보상도 크다면, 현재 Q값을 해당 방향으로 조금 끌어 올림.
        q_table[state + (action,)] += alpha * (reward + gammer * best_next_q - q_table[state + (action,)])  # 벨만방정식 적용

        # 에이전트가 한 스탭을 마친 후, 다음 스탭 준비를 위한 상태 갱신 및 기록하기.
        state = next_state
        obs = next_obs
        total_reward += reward
        trajectory.append(obs.copy())
        if done:
            break       # 에피소드 종료
    
    reward_list.append(total_reward)    #에피소드에서 받은 최종 결과를 리워드 리스트에 저장.

    if total_reward > best_reward:
        best_reward = total_reward
        print(f'Episode {ep} : reward improved to {total_reward}')
    
    if ep % 10 == 0:    # 10회당 1번씩만 저장.
        trajectories.append(trajectory)
    
    if epsilon > epsilon_min:   # 엡실론 값이 점점 감소 - 학습이 진행됨에 따라 무작위 행동을 줄이고 점점 더 최적 정책에 집중하기 위한 전략.
        epsilon *= epsilon_decay

### 학습 곡선 보상 그래프 시각화 해보기.
plt.figure(figsize=(10, 4))
plt.plot(reward_list, label='Episode Reward')
plt.title('Episode Rewards 0ver time')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


### 카트폴 시뮬레이션을 애니메이션화 하기
flat_state = []     #여러 에피소드의 궤적을 한 줄(flat)로 펼쳐서 저장할 리스트 타입 번수.
episode_labels = []
episode_numbers = list(range(0, episodes, 10))  # 10회마다 저장
#print(episode_numbers)

#데이터 평탄화
for i , traj in enumerate(trajectories):
    #print(len(traj))
    #print(traj)
    flat_state.extend(traj)
    episode_labels.extend([episode_numbers[i]] * len(traj))
#print(flat_state)
#print(episode_labels)

frame_count = len(flat_state)   # 에니메이션에 사용된 총 프레임 수
#print(frame_count)              # 8364, 8364장의 그림으로 에니메이션
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title('Cart simulation')
ax.set_xlabel('Cart position')
ax.set_ylabel('Height')
cart_width = 0.4
cart_height = 0.2
cart_y = 0.0
cart_rect = Rectangle((0, 0), cart_width, cart_height, color='black')
ax.add_patch(cart_rect)

# pole 그리기 - Line2D
pole_len = 1.0
line_list = ax.plot([], [], 'r-', lw=4)
pole_line = line_list[0]

episode_text = ax.text(0.05, 1.4, '', transform=ax.transData, color='blue')

def update(frame):
    x = flat_state[frame][0]        # 카트 위치, 카트 속도, 막대 각도, 막대각 속도
    #print('x: ', x)
    theta = flat_state[frame][2]    # 현재 프레임의 막대 각도(radian 단위)를 세타에 저장.
    ep_num = episode_labels[frame]
    cart_rect.set_xy((x - cart_width / 2, cart_y))
    # pole 끝 좌표 계산
    x_start = x
    y_start = cart_y + cart_height
    x_end = x_start + pole_len * np.sin(theta)
    y_end = y_start + pole_len * np.cos(theta)
    pole_line.set_data([x_start, x_end], [y_start, y_end])

    episode_text.set_text(f'Episode : {ep_num}')
    return cart_rect, pole_line, episode_text

ani = FuncAnimation(fig, update, frames=frame_count, interval=50, repeat=False)
plt.close(fig)      #그림 중복 방지를 위한 close
ani.save("animation.mp4", writer="ffmpeg")