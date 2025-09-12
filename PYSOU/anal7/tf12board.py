"""
다중선형회귀 모델(다항선형회귀 아님!) 만들기. + TensorBoard 사용해보기
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from sklearn.metrics import r2_score
import shutil, os, datetime as dt

#작년의 5명의 3회 시험 점수로 모의고사 후 본고사 점수 데이터로 학습 후, 새 데이터로 점수 예측
xData = np.array([[70, 85, 80], [71, 89, 78], [50, 80, 60], [66, 30, 60], [50, 25, 10]])
yData = np.array([73, 82, 72, 57, 34])

#Sequential API 사용
print('1) Sequential API 사용')
model = Sequential()
model.add(Input(shape=((3,))))
model.add(Dense(units=8, activation='relu', name='a'))  #name : 텐서보드에 띄울 이름.
model.add(Dense(units=4, activation='relu', name='b'))
model.add(Dense(units=1, activation='linear', name='c'))
print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 a (Dense)                   (None, 8)                 32

 b (Dense)                   (None, 4)                 36

 c (Dense)                   (None, 1)                 5

=================================================================
Total params: 73
Trainable params: 73
Non-trainable params: 0
"""

opti = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model.fit(xData, yData, batch_size=1, epochs=50, verbose=2)

#plt.plot(history.history['loss'])
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.show()
#plt.close()

loss_metrics = model.evaluate(x=xData, y=yData)
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(yData, model.predict(xData)))   #설명력 :  0.886995255947113

from keras.models import Model
from keras.callbacks import TensorBoard

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


#Functional API 사용
print('2) Functional API')
xData = np.array([[70, 85, 80], [71, 89, 78], [50, 80, 60], [66, 30, 60], [50, 25, 10]])
yData = np.array([73, 82, 72, 57, 34])

inputs = Input(shape=(3,))
h1 = Dense(units=8, activation='relu', name='a')(inputs)
h2 = Dense(units=8, activation='relu', name='b')(h1)
outputs = Dense(units=1, activation='linear', name='c')(h2)

model2 = Model(inputs, outputs, name='linear_model')

# TensorBoard ------
BASE = "logs"           #기본 로그 저장 디렉토리의 이름
shutil.rmtree(BASE, ignore_errors=True)     #해당 디렉토리 전체를 삭제, 에러 메시지 무시
RUN = os.path.join(BASE, 'test')            #logs\test\train 형태로 파일이 생성됨.
os.makedirs(RUN, exist_ok=True)

tb = TensorBoard(log_dir=RUN, histogram_freq=1, write_graph=True)     #매 epoch마다 가중치 히스토그램을 만듬.
# ------------------

opti = optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])
model2.fit(xData, yData, batch_size=1, epochs=50, verbose=2, callbacks=[tb])

loss_metrics = model2.evaluate(x=xData, y=yData)
print('loss_metrics : ', loss_metrics)
print('설명력 : ', r2_score(yData, model2.predict(xData)))


from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
