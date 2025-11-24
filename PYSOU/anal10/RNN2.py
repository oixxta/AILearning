"""
LSTM 이해 목적 : 다음 숫자 예측

LSTM은 simpleRNN의 단점을 보완한 모델
"""

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Input, Dense

# 데이터와 라벨 정의
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 50, 60, 70])


# LSMT에 맞게 3D 입력 형태로 변환시키기 (sample, time_steps, features)
x = x.reshape((8, 3, 1))
print(x.shape, ' ', y.shape)

# 모델 정의하기
model = Sequential()
model.add(Input(shape=(3, 1)))      # 시계열 데이터 3개, 라벨은 1개
model.add(LSTM(10, activation='relu'))
model.add(Dense(1, activation='linear'))
print(model.summary())

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 1)                 11

=================================================================
Total params: 491
Trainable params: 491
Non-trainable params: 0
_________________________________________________________________
"""
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=3, mode='auto')
model.fit(x, y, epochs=50, batch_size=4, verbose=2, callbacks=[es])        # 8개의 셈플을 4개씩 나눠서 2 번에 걸쳐서 10번 돌림.

print('예측값 : ', model.predict(x))
print('실제값 : ', y)

# 새로운 값으로 예측
x_input = np.array([25, 35, 47])
x_input = x_input.reshape((1, 3, 1))
pred = model.predict(x_input)
print(pred.flatten())