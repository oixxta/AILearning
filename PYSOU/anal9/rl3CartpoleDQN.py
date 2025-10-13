"""
DQL으로 카트플 만들기 학습

딥러닝을 쓰기에 카트폴 구성 필요 없음.
핵심 구성 요소 : 
Q-Network, 
Target Network (-Q값 계산용 네트워크를 하나 더 구성, 일정 주기마다 Q-Network의 가중치를 복사해서 사용함.),
Experience Replay (매 번의 경험을 (state, action, reweard, save), done 형태로 저장 후 
무작위로 셈플링하여 학습에 사용 : 상관관계 제거 및 다양성 확보를 위해)
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
#from IPython.display import HTML
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque       #경험 리플레이 저쟝용 자료구조.
import random
from tensorflow import keras


### 환경 설정
# 강화학습은
# agent는 state를 받아 action을 선택하고, 그에 따른 reward를 받으며 학습한다.
# 학습 목표 : agent가 총 보상을 최대화 하는 방향으로 policy를 학습한다.
env = gym.make('CartPole-v1')
num_actions = env.action_space.n                #카트폴이기에 좌우 움직임만 action
state_dim = env.observation_space.shape[0]
print(num_actions)      # 2
print(state_dim)        # 4 : 카트 위치, 카트 속도, 막대 각도, 막대 각도 속도

### DQN 모델 정의
def create_model():
    model = Sequential([
        Input(shape=(state_dim, )),
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=num_actions, activation='linear'), #출력층은 정량적인 값이 나오기에 리니어를 사용.
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mse']) 
    #목표값(target) : reward + γ * maxQ(state1, action1)
    #예측값(prediction) : Q(state, action)
    #예측값과 목표값의 차이를 최소화 하는 것이 목적.
    return model

model = create_model()  # 주 네트워크 : 학습을 직접 수행하는 신경망
target_model = create_model()   # 타겟 네트워크 : 학습중인 모델과는 별개로 유지되는 Q-network
# Q-learning의 안정성 향상을 위해 사용하는 '고정된 Q값 계산용 네트위크'
target_model.set_weights(model.get_weights())


### 하이퍼 파라미터
gammer = 0.99
epsilon = 1.0   # 탐험률 100%
epsilon_min = 0.05  # 5퍼센트 확률로 여전히 탐험
epsilon_decay = 0.995
batch_size = 64 # 경험 리플레이에서 랜덤하게 꺼내서 학습에 사용할 셈플의 숫자 32~128 사이가 권장됨.
# 경험 리플레이 버퍼 : 양방향 큐 자료구조(FiFO)로, append(), popleft() 사용함.
memory = deque(maxlen=5000)  # 경험 재사용.
episodes = 50    # 200 ~ 500이 권장됨.
# Target Q-Network 갱신주기
update_target_every = 5
reward_list = []


### 학습 루핑
for ep in range(episodes):
    state, info = env.reset()   # 에피소드 시작 전 초기화 시키기
    total_reward = 0            # 에피소드마다 받는 보상 저장
    done = False                # 에피소드 종료 조건 (막대가 너무 기울거나 제한시간이 끝날 경우)

    while not done:         # 막대가 너어진 경우나, 시간 초과인 경우 반복 종료.
        state_input = np.reshape(state, [1, state_dim]) # DQN에서 state를 원하기 때문에 필요(신경망 입력 형식에 맞게 변형).

        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
        
        next_state, reward, terminated, truncated, _ = env.step(action) #선택한 액션을 환경에 적용, 해당하는 값 출력.
        done = terminated or truncated

        modified_reward = reward if not done else -10
        memory.append((state, action, modified_reward, next_state, done))
        state = next_state
        total_reward += reward

        # 학습 : 일정 수 이상 경험이 쌓이면 학습을 시작함. 벨만 방정식 기반의 Q값 갱신
        if len(memory) >= batch_size:   #배치사이즈보다 크거나 같아야 학습 시작
            minibatch = random.sample(memory, batch_size)
            states, targets = [], []    # 상태 입력값, Q값 target을 저장할 배열

            for s, a, r, s_next, d in minibatch:
                s_input = np.reshape(s, [1, state_dim]) #[1, state_dim] : 1차원을 2차원화
                s_next_input = np.reshape(s_next, [1, state_dim])
                target = model.predict(s_input, verbose=0)[0]   # target : [Q(s, 0), Q(s, 1)]
                #print(target)
                if d:
                    target[a] = r   # 종료 상태면 미래 보상 없음.
                else:
                    t_next = target_model.predict(s_next_input) # Q(s,a)는 즉시 보상 + 미래 최대 Q값
                    target[a] = r + gammer * np.max(t_next) # Target을 이용해 신경망을 업데이트
                states.append(s)        # 입력 데이터 리스트에 현재 상태 저장
                targets.append(target)  # 정답 Q값 백터에 저장.
            
            #print('states : ', np.array(states), ':', np.array(targets))
            model.fit(np.array(states), np.array(targets), epochs=1, verbose=1)      #우측이 레이블
            # 여기까지 : 리플레이 버퍼에서 무작위로 batch 꺼냄 -> 샘플에 대해서 Q(s, a)를 갱신함 -> 전체 states, targets를 모아 model.fit()
    
    reward_list.append(total_reward)    # 한 에피소드 동안 받은 보상 누적

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min) # 더 이상 줄어들지 않을 최소 탐험 비율을 지정함.
    
    # 타겟 모델 갱신 (target network)
    if ep % update_target_every == 0:
        target_model.set_weights(model.get_weights())
    
    if ep % 10 == 0:
        print(f'Episode {ep} : Reward = {total_reward:.1f}, Epsilon = {epsilon:.3f}')


### 보상 시각화
plt.figure(figsize=(10, 4))
data = np.array([1, 2, 3, 4, 5])
window_size = 3
avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
print(avg)

#이동 평균 계산 (노이즈 제거용)
def moving_average(data, window_size = 10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

plt.plot(reward_list, label = 'Reward per Episode')
plt.plot(moving_average(reward_list), label='Moving avg', color = 'red')
plt.title('DQN Cartpole Reward')
plt.xlabel('episode')
plt.ylabel('reward')
plt.grid(True)
plt.tight_layout()
plt.show()


### 모델 저장
model.save('dqn_model.keras')
print('model save complete! : dqn_model.keras')


### 에니메이션 저장
env = gym.make('CartPole-v1', render_model = None)
model = keras.models.load_model('dqn_model.keras')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

flat_state = []
episode_label = []

state, _ = env.reset()
done = False
ep_num = 0

while not done:
    flat_state.append(state.copy())
    episode_label.append(ep_num)
    state_input = np.reshape(state, [1, state_dim])
    q_values = model.predict(state_input, verbose = 0)
    action = np.argmax(q_values[0])
    next_state, reward, terminated, truncated, _ = env.step(action)
    state = next_state
    done = terminated or truncated
env.close()

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
    ep_num = episode_label[frame]
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
ani.save("animation2.mp4", writer="ffmpeg")