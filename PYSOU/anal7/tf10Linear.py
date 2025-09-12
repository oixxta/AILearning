"""

"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input              #케라스의 최 하단은 무조건 덴스
from keras import optimizers
import numpy as np

#데이터 수집 및 가공, 상관계수 확인.
xData = np.array([1., 2., 3., 4., 5.]).reshape(-1, 1)  #실수 타입이 든 배열
#덴스 측은 출력을 1차원으로 해도 2차원 배열을 기대함.
yData = np.array([1.2, 2.0, 3.0, 3.5, 5.3]).reshape(-1, 1)
print('상관계수 : ', np.corrcoef(xData.ravel(), yData.ravel())) #revel을 안쓰면 차원이 안맞아서 다 nan으로 뜸. 주의.
#강한 양의 상관관계를 가지는 것을 확인.


#모델 만들기
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(units=32, activation='relu'))   #히든레이어의 엑티베이션은 무조건 relu로
model.add(Dense(units=32, activation='relu'))   #히든레이어의 엑티베이션은 무조건 relu로
model.add(Dense(units=1, activation='linear'))
print(model.summary())

model.compile(optimizer='sgd', loss='mse', metrics=['mse']) #분류 : 엔트로피, 회귀 : MSE 사용.
model.fit(xData, yData, batch_size=1, epochs=10, verbose=1, shuffle=True)
print(model.evaluate(xData, yData))

"""
Epoch 1/10
5/5 [==============================] - 1s 1ms/step - loss: 30.3287 - mse: 30.3287
Epoch 2/10
5/5 [==============================] - 0s 2ms/step - loss: 1.7425 - mse: 1.7425
Epoch 3/10
5/5 [==============================] - 0s 1ms/step - loss: 0.2958 - mse: 0.2958
Epoch 4/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1796 - mse: 0.1796
Epoch 5/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1588 - mse: 0.1588
Epoch 6/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1658 - mse: 0.1658
Epoch 7/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1566 - mse: 0.1566
Epoch 8/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1521 - mse: 0.1521
Epoch 9/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1479 - mse: 0.1479
Epoch 10/10
5/5 [==============================] - 0s 1ms/step - loss: 0.1463 - mse: 0.1463
1/1 [==============================] - 0s 139ms/step - loss: 0.1034 - mse: 0.1034
[0.10339510440826416, 0.10339510440826416]
"""

pred = model.predict(xData) #예측
print('예측값 : ', pred.ravel())    #예측값 :  [1.248743  2.1788616 3.1089802 4.0390987 4.9692173]
print('실제값 : ', yData.ravel())   #실제값 :  [1.2       2.        3.        3.5       5.3      ]


#결정계수 보기
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(yData, pred))   #설명력 :  0.9516341856757208, 95%


#시각화 하기
import matplotlib.pyplot as plt
plt.scatter(xData, yData, color='r', marker='o', label='real')
plt.plot(xData, pred, 'b--', label='pred')
plt.show()
plt.close()

