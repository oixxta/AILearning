"""
강화학습을 활용한 주식 트레이딩 구현

prices.csv 파일 사용.

주식 트레이딩은 주식을 단기간에 사고팔아 수익을 내는 행위.
투자자의 감정을 배제하고 자동화된 시스템으로 거래하는 방식.

트레이딩은 일정 기간 내 주식을 사고팔아 수익을 얻는 행위를 뜻한다고 했는데 이 기간은 수 분에서 수일, 
혹은 수주에 불과할 수 있으며, 매매 빈도가 잦음.


강화학습(DQN)으로 단일 종목 일봉 수익률에 대해 롱/숏/현금 포지션을 
선택하며 누적 PnL을 최대화하려는, 가장 단순한 실습용 트레이딩 에이전트.

강화학습 에이전트가 매일 어떤 행동을 선택하느냐를 뜻함.

  포지션         의미                행동 코드     설명
  롱(Long)      매수 후 상승에 베팅   +1           “내일 오를 거야” → 주식을 들고 있음
  숏(Short)     공매도, 하락에 베팅   -1           “내일 떨어질 거야” → 주식을 빌려팔고 나중에 다시 삼
  현금(Cash)    아무것도 안 함         0           포지션 없이 대기

즉, 에이전트는 매일 아침 “오늘은 롱 할까, 숏 할까, 그냥 쉴까?” 를 결정하는 문제를 학습하는 것.


참고 : 샤프비율(Sharpe Ratio)란? 수익률의 ‘위험 대비 성과’를 나타내는 지표.
실제 금융에서는 1.0 이상이면 “쓸 만한 전략”, 2.0 이상이면 “훌륭한 전략”으로 평가.
"""
# 환경 : 20일 기준
# 행동 : -1, 0, +1, (Short, Cash, Long)
# 보상 : (이전 포지션 * 금일 수익률) - 비용(거래 변경량 * bps)  # 100bps = 1%
# 포지션을 바꾸면, 거래 비용이 든다(cost_bps)
# Step : 보상 계산 후 시점을 하루 전진, 다음 관측을 반환함
# DQN 트레이딩 초간단 실습

import math, random
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf


def load_returns(csv_path:str):
    df = pd.read_csv(csv_path)
    close = df['Close'].astype(float).values
    ret = np.zeros_like(close, dtype=np.float32)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1]     #일별 수익률 계산
    return ret

# 환경
class TradingEnv:
    def __init__(self, returns:np.ndarray, window=20, cost_dps = 10.0):        #수익률, 관측 빈도, 거래 미용 데이터를 받음
        assert len(returns) > window + 1, '데이터가 너무 적음.'
        self.rets_all = returns.astype(np.float32)
        self.window = window
        self.cost = cost_dps / 10000.0 # bps(1 / 100 bp)
        self.reset()
    
    @property                                 #메서드이지만 맴버 필드처럼 부르게 하는 코드
    def obs_dim(self):return self.window + 1  #관측 차원

    @property    
    def n_actions(self):return 3

    def reset(self):
        self.t = self.window
        self.pos = 0
        return self._obs()
    
    def _obs(self):
        window = self.rets_all[self.t - self.window:self.t] #직전 윈도우 일의 수익률 슬라이스\
        return np.concatenate([window, [float(self.pos)]]).astype(np.float32) #수익률들 + 햔재 포지션을 결합

    def step(self, action:int):
        new_pos = [-1, 0, 1][action]
        trade_cost = self.cost * abs(new_pos - self.pos)    #거래 비용(포지션 변화량)
        reward = self.pos * self.rets_all[self.t] - trade_cost  #보상(이전 포지션 * 금일 수익률 - 거래 비용)
        self.pos = new_pos
        self.t += 1 # 하루 전진
        done = (self.t >= len(self.rets_all) - 1)   # 마지막 전날까지 진행하면 에피소드 종료
        return self._obs(), float(reward), done     # 다음 관측, 보상, 종료 여부 반환
    
# Q-Network
def build_qnet_seq(obs_dim, n_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(obs_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions),
    ])
    return model

# DQN Agent
class Replay:
    def __init__(self, cap=20000):self.buf = deque(maxlen=cap)  #transcation(전이)
    def __len__(self): return len(self.buf)
    def push(self, *tr):self.buf.append(tr) # 전이(s, a, r, s', done) 하나를 버퍼에게 주기
    def sample(self, n):
        s = random.sample(self.buf, n)    #버퍼에서 n개 무작위 추출
        s, a, r, ns, d = zip(*s)    #전이 튜플을 각 배열로 분리
        return (np.array(s, np.float32),
                np.array(a, np.float32),
                np.array(r, np.float32),
                np.array(ns, np.float32),
                np.array(d, np.float32))
    
# DQN Agent(기본형)
class DQN:
    def __init__(self, obs_dim, n_actions, lr = 3e-4, gammer=0.99, batch=128):
        self.q = build_qnet_seq(obs_dim, n_actions)     #메인 네트워크 생성
        self.tgt = build_qnet_seq(obs_dim, n_actions)   #타겟 네트워크
        self.tgt.set_weights(self.q.get_weights())       #타겟 가중치를 메인 네트워크와 같게 복제
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.gammer, self.batch = gammer, batch
        self.buf = Replay()
        self.loss_fn = tf.keras.losses.Huber()          #후버 손실 : mse보다 이상치에 덜 민감함.
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9995
        self.n_actions = n_actions

    def act(self, obs):         #행동 선택 : 탐험 혹은 이용
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        qv = self.q(obs[None, : ], training=False).numpy()[0]   #Q(s, ) 계산
        return int(np.argmax(qv))       # q값이 최대인 행동(활용)
    

    def update(self):           #파라미터 갱신 실시
        if len(self.buf) < self.batch:return
        s, a, r, ns, d = self.buf.sample(self.batch)    # 미니배치 전이 샘플링

        a_oh = tf.one_hot(a, self.n_actions)    # 행동 인덱스를 원핫 백터로 변환하기 (Q(s,)에서 Q(s,a) 추출용)
        with tf.GradientTape() as tape:      # 경사하강법으로 미분
            q_sa = tf.reduce_sum(self.q(s) * a_oh, axis = 1)    # 원핫의 내적
            q_next = tf.reduce_max(self.tgt(ns), axis=1)        # 타겟 네트워크로 다음 상태의 최대 Q값
            y = r + (1 - d) * self.gammer * q_next
            loss = self.loss_fn(y, q_sa)                # 손실 = Huber(y, Q(s, a))
        
        q = tape.gradient(loss, self.q.trainable_variables)     # 온라인 네트워크에 대한 그래디언트 값 계산하기.
        self.opt.apply_gradients(zip(q, self.trainable_variables))  # 경사 하강 스텝 저용하기
        self.tgt.set_weights(self.q.get_weights())
        self.eps = max(self.eps * self.eps_decay, self.eps_min)     # 탐험 비율을 점진적으로 수정해 나감.


def train(csv_path='prices.csv', window=20, cost_bps=10.0, episodes=5):
    rets = load_returns(csv_path)
    # print(rets)
    env = TradingEnv(rets, window, cost_bps)
    agent = DQN(env.obs_dim, env.n_actions)
    equity = []     # 누적 PnL 추적 리스트(성과 지표용)

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done, ep_pnl = False, 0.0
        while not done: # 데이터 끝까지 하루씩 진행
            act = agent.act(obs)
            nobs, r, done = env.step(act)   # 다음 관측, 보상, 종료를 반환받음.
            agent.buf.push(obs, act, r, nobs, float(done))
            agent.update()
            obs = nobs
            ep_pnl += r     # 에피소드 pnl을 누적함.
            equity.append(ep_pnl)
        print(f'[EP {ep} / {episodes}] PnL = {ep_pnl:.4f}, eps = {agent.eps:.3f}')
    
    #간단한 요약 결과 출력
    equity = np.array(equity)
    daily_ret = np.diff(equity, prepend=0)  # 일별 PnL 증분(=보상 시퀀스) 추정
    sharp = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252) # 샤프 비율 근사 : (일별 수익룰 평균 / 일별 수익률 표준 편차) * root252
    print('요약 결과')
    print(f'Final PnL : {equity[-1]:.6f}')
    print(f'Sharp(위험 대비 수입 척도) : {sharp:.3f}')

    agent.q.save('dqn_model.keras') # 모델을 저장


if __name__ == '__main__':
    train(csv_path = 'prices.csv', window=20, cost_bps=10.0, episodes=5)











