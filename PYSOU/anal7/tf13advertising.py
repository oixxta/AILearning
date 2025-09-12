"""


"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale, RobustScaler

#StandardScaler : 표준편차 이용, minmax_scale : 최소 최댓값 이용, RobustScaler : 중앙값과 사분위 값을 씀. 가장 아웃라이어에 둔감함.

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv')
print(data.head(2))
del data['no']
print(data.head(2))

fData = data[['tv', 'radio', 'newspaper']]
lData = data[['sales']]
print(fData[:2])
print(lData[:2])
"""
      tv  radio  newspaper
0  230.1   37.8       69.2
1   44.5   39.3       45.1

   sales
0   22.1
1   10.4
"""

#정규화 하기
#scaler = MinMaxScaler(feature_range=(0, 1))
#feData = scaler.fit_transform(fData)
#print(feData[:3])

feData = minmax_scale(fData, axis=0, copy=True) #copy=True : 원본이 보존됨. False로 할 경우 원본이 갱신됨.
print(feData[:3])
"""
[[0.77578627 0.76209677 0.60598065]
 [0.1481231  0.79233871 0.39401935]
 [0.0557998  0.92540323 0.60686016]]
"""

#훈련/테스트 데이터 나누기
xTrain, xTest, yTrain, yTest = train_test_split(feData, lData, test_size=0.3, random_state=123)

#모델 만들기
model = Sequential()
model.add(Input(shape=(3,)))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 16)                64

 dense_1 (Dense)             (None, 8)                 136

 dense_2 (Dense)             (None, 1)                 9

=================================================================
Total params: 209
Trainable params: 209
Non-trainable params: 0
_________________________________________________________________
"""

#만든 모델 시각자료 만들어서 저장하기
tf.keras.utils.plot_model(model, 'tf13.png')

history = model.fit(xTrain, yTrain, epochs=100, batch_size=32, verbose=2 ,validation_split=0.2) #validation_data = (x_vali, y_vali)

#모델 평가하기
loss = model.evaluate(xTest, yTest, verbose=0)
print('loss : ', loss[0])  #[5.439108371734619]

#history 값 확인하기
#print('history : ', history.history)
print('loss in history : ', history.history['loss'])
print('mse : ', history.history['mse'])
print('val_loss : ', history.history['val_loss'])   #validation_split에서 가져온 값
print('val_mse : ', history.history['val_mse'])     #validation_split에서 가져온 값

#loss 시각화
plt.plot(history.history['loss'], color='blue', label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.close()

#r2score 확인해 보기
print('r2Score : ', r2_score(yTest, model.predict(xTest)))  #r2Score :  0.7125484943389893
#독립변수가 종속변수의 연관을 충분히 설명함.

