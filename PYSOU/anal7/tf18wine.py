"""
wind data set으로 red, white 와인 분류 모델 작성하기
Deeplearning 사용, earlyStoping 사용, modelcheckpoint 사용, Dropout 사용
"""
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#데이터 가져오기
wdf = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv', header=None)
print(wdf.head(5))
print(wdf.info())
print(wdf.iloc[:, 12].unique()) #12번째 칼럼은 0 아니면 1, 1 : red, 0 : white
print(len(wdf[wdf.iloc[:, 12] == 0]))   #화이트 와인 4898개
print(len(wdf[wdf.iloc[:, 12] == 1]))   #레드 와인 1599개


#데이터 label이랑 feature 나누기
dataset = wdf.values
x = dataset[:, 0:12]
y = dataset[:, -1]
np.set_printoptions(suppress=True)  #과학적 표기법 끄기
print(x[:3])    # feature
print(y[:3])    # label


#학습데이터 분리하기
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=12, shuffle=True, stratify=y)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
print(xTrain[:3])
print(yTrain[:3])


#모델 생성하기 : Sequential
model = Sequential()
model.add(Input(shape=(12,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))        #드랍아웃, 과적합 방지용
model.add(BatchNormalization())     #배치 정규화, 역전파시 기울기 소실 또는 폭주 방지. CNN에서 효과적.
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=0.1))
model.add(BatchNormalization()) 
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) #Dense 만이 regression과 classification을 둘 다 할수 있음.
print(model.summary())
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 32)                416

 dropout (Dropout)           (None, 32)                0

 batch_normalization (BatchN  (None, 32)               128
 ormalization)

 dense_1 (Dense)             (None, 16)                528

 dropout_1 (Dropout)         (None, 16)                0

 batch_normalization_1 (Batc  (None, 16)               64
 hNormalization)

 dense_2 (Dense)             (None, 8)                 136

 dense_3 (Dense)             (None, 1)                 9

=================================================================
Total params: 1,281
Trainable params: 1,185
Non-trainable params: 96
_________________________________________________________________
"""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
loss, acc = model.evaluate(xTrain, yTrain, verbose=0)    #fit을 하기 전에 모델의 score를 확인할 수 있음.
print('훈련 전 모델 정확도 : {:5.2f}%'.format(100 * acc))   #훈련 전 모델 정확도 : 75.57%


#모델 체크포인트 사용 : 모델 외부파일로 저장하기
MODEL_DIR = './model/' #모델 저장 폴더 설정
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = 'model/{epoch:02d}_{val_loss:.4f}.keras'
#모델 학습 과정에서 특정 기준에 따라 자동으로 모델을 저장하는 callback
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='auto', save_best_only=True)    #save_best_only : 가장 좋은 성능일때만 저장함.


#모델 학습하기 : 얼리스탑도 사용
earlyStop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(xTrain, yTrain, validation_split=0.2, epochs=1000, batch_size=64, callbacks=[earlyStop, chkpoint], verbose=2)
loss, acc = model.evaluate(xTest, yTest, batch_size=64, verbose=0)
print('훈련 후 모델 정확도 : {:5.2f}%'.format(100 * acc))   #훈련 후 모델 정확도 : 96.36%


#시각화 해보기
epochLen = np.arange(len(history.epoch))
plt.plot(epochLen, history.history['val_loss'], label='val_loss')
plt.plot(epochLen, history.history['loss'], label='loss', c='red')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
plt.close()

plt.plot(epochLen, history.history['val_accuracy'], label='val_accuracy')
plt.plot(epochLen, history.history['accuracy'], label='accuracy', c='blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
plt.close()


#저장된 가장 잘 나온 모델로 예측해보기
from keras.models import load_model
model = load_model('model/30_0.0680.keras')     #경로 + 파일명을 스트링 타입으로
newData = xTest[:5, :]
print(newData)
pred = model.predict(newData)
print('예측 결과 : ', np.where(pred >= 0.5, 1, 0).ravel())