"""
글자 단위 토큰 생성 후, 문자열 생성 모델 작성.
자소(grapheme)

cher_ex.txt 파일을 활용해 연습.
"""

import os, sys, random, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 각종 랜덤 시드들 고정
tf.random.set_seed(42)  
np.random.seed(42)
random.seed(42)

# 읽어올 파일 가져오기
fileName = 'char_ex.txt'
with open(fileName, encoding='utf-8') as f:
    et = f.read().lower()

print(et[:300] if len(et) > 300 else et) # 읽어올 파일의 글자 수가 300보다 많으면 300개만 읽어옴.

# 문자(character) 단위 어휘집 생성함.
chars = sorted(list(set(et)))   # 고유 문자만 뽑아서 정렬 실시함.
print(chars)                    # 사전 순으로 어센딩 소트가 완료됨.

char_to_int = {c:i for i, c in enumerate(chars)}
print(char_to_int)
int_to_char = {i:c for i, c in enumerate(chars)}
print(int_to_char)
n_chars = len(et)
print('전체 문자 수 : ', n_chars)
n_vocab = len(chars)
print('전체 어휘 수 : ', n_vocab)

# 시퀀스 구성하기
seq_length = 10     # 입력 윈도우 길이 (ex : 10글자 -> 다음 1글자 예측)
dataX, dataY = [], []

for i in range(0, n_chars - seq_length, 1):     # 한 글자씩 슬라이딩 윈도우
    seq_in = et[i:i + seq_length]               # 입력 문자열
    seq_out = et[i + seq_length]                # 다음 예측 글자
    dataX.append([char_to_int[ch] for ch in seq_in])    # 입력을 숫자 시퀀스로 전환 후 기억.
    dataY.append(char_to_int[seq_out])          # 레이블 기억

print(dataX)
print(dataY)    # [0, 3, 10, 4, 0, 13, 9, 12, 0, 9, 6, 3, 13, 2]


N = len(dataX)  # 전체 학습 샘플(시퀀스)의 갯수. ex:1000글자 텍스트, seq_length=10이면, 990개의 시퀀스가 생김.
print('dataX의 행렬 유형 수 : ', N)     # 14개

if N == 0:
    raise ValueError("입력 데이터가 너무 적음. 시퀀스 처리 불가능.")


# 입출력 원 핫 인코딩 처리
x = to_categorical(dataX, num_classes=n_vocab)
y = to_categorical(dataY, num_classes=n_vocab)
#print(x)


# 모델 만들기 (네트워크 구성)
model = Sequential([
    Input(shape=(seq_length, n_vocab)),      # 입력 : (시퀀스 길이, 문자 종류의 수)
    LSTM(128, return_sequences=True),   # many to many. 모든 타임 스탭을 출력함.
    Dropout(0.2),
    LSTM(128),                          # 마지막 타임 스탭만을 출력함.
    Dropout(0.2),
    Dense(n_vocab, activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

chkpoint_pass = 'mymodel/rnn6model.keras'
os.makedirs(os.path.dirname(chkpoint_pass), exist_ok=True)
chkpoint = ModelCheckpoint(chkpoint_pass, monitor='loss', save_best_only=True, mode='min', verbose=0)
es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
batch_size = min(8, max(1, N // 2))
history = model.fit(x, y, epochs=500, batch_size=batch_size, verbose=2, callbacks=[es, chkpoint])


"""
# 학습 곡선 시각화 하기.
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
acc_ax.plot(history.history['accuracy'], label='train loss')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], label='train accuracy')
loss_ax.set_ylabel('accuracy')
loss_ax.legend(loc='lower left')
plt.tight_layout()
plt.show()
plt.close()
"""

# 모델이 예측한 확률 분포에서 다양성(temperature)과 top-k를 적용해 글자의 인덱스를 무작위로 선택하는 함수.
def sample_with_temperFunc(probs, temparature=0.8, top_k=5):    #상위 후보 5개만 정규화에 참여.
    p = np.array(probs, dtype=np.float64)

    #k = 3
    #arr = np.array([7, 2, 9, 4, 1])
    #idx = np.argpartition(-arr, k)[:k]   #부분자료 추출 후, 정렬을 빠르게 하는 함수. 음수로 반전 시킴
    #print(idx)                           #[2 0 3], 인덱스 값.

    # 상위 k개 확률만 남기기
    if top_k is not None and top_k > 0 and top_k < len(p):
        idx = np.argpartition(p, -top_k)[-top_k]    # 상위 k 인덱스 선택
        mask = np.zeros_like(p)   # 동일 크기의 배열 (0 기억) 준비.
        mask[idx] = p[idx]        # 선택된 k개 위치만 원래 확률 유지. 나머지는 0
        p = mask                  # 확률값이 낮은 후보는 제외함.
        # ex : [0.7, 0.2, 0.1] -> top_k = 2 라면, [0.7, 0.2, 0.0]

    # temperature (다양성) 조정 - 분포를 늘리거나 줄임.
    p = np.log(p + 1e-9) / max(temparature, 1e-9)
    p = np.exp(p)       # 다시 지수화 시킴. log 확률을 원래 확률 공간으로 복원.
    p = p / p.sum()     # 확률 재정규화 작업을 함. 확률의 총 합이 1이 될 수 있게.

    # 확률 p에 따라 인덱스 하나를 무작위로 선택해 선택된 인덱스를 정수로 반환.
    return int(np.random.choice(len(p), p=p))

# 문장 생성
start = np.random.randint(0, N - 1)     # 랜덤 시작 인덱스
pattern = list(dataX[start])
seed_text = ''.join(int_to_char[v] for v in pattern)    # ex:['h', 'e', 'l'] -> 'hel'
print('seed: ')
print(f"\"{seed_text}\"")

step = 500  # 생성할 문자 수
temperature = 0.8       # 0에 가까울수록 보수적(예측 가능), 멀어지면 창의적 - 다양성이 증가됨.
top_k = 5               # 확률 상위 후보 갯수

generated = []
for _ in range(step):
    x = to_categorical([pattern], num_classes=n_vocab)
    probs = model.predict(x, verbose=0)[0]      # 다음 문자의 확률 예측.
    idx = sample_with_temperFunc(probs=probs, temparature=temperature, top_k=top_k)

    ch = int_to_char[idx]
    generated.append(ch)
    pattern.append(idx)     # 입력 시퀀스 갱신함.
    pattern = pattern[1:]   # 시퀀스 슬라이싱(앞 글자 제거)

gen_text = ''.join(generated)
print("\n[Generated]")
print(gen_text)
print("Done")


