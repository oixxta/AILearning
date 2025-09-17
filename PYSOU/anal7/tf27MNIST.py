"""
숫자 이미지 구분하기 (0 ~ 9, 다항분류 모델)

MNIST 데이터 세트 사용, CNN 미사용
"""
import tensorflow as tf
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import to_categorical

# MNIST dataset(숫자 이미지) 가져오기
(xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)
print(xTrain[0])    #0번째 feature
print(yTrain[0])    #0번째 label
#plt.imshow(xTrain[0], cmap='gray')     #직접 확인해 보기
#plt.show()

# x 데이터들 정규화 및 실수화 하기.
xTrain = xTrain.reshape(-1, 784).astype('float32')  # 28 by 28 => 784 열로 변경
xTest = xTest.reshape(-1, 784).astype('float32')
xTrain = xTrain / 255.0     #픽셀 값들을 0 ~ 1 범위로 정규화
xTest = xTest / 255.0

# y 데이터들 원 핫 인코딩 적용하기.(출력층 활성화 함수를 softmax를 쓰기 때문.)
yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)

# validation data 만들기
xVali = xTrain[50000:60000]
yVali = yTrain[50000:60000]
xTrain = xTrain[0:50000]
yTrain = yTrain[0:50000]
print(xVali.shape, xTrain.shape)    #(10000, 784) (50000, 784)

# 신경망 모델 만들기 및 학습 : Sequential
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(xTrain, yTrain, epochs=10, batch_size=128, validation_data=(xVali, yVali), verbose=2)

# 모델 평가 및 히스토리 시각화
print('loss : ', history.history['loss'])
print('val_loss : ', history.history['val_loss'])
print('acc : ', history.history['acc'])
print('val_acc : ', history.history['val_acc'])

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], label='loss')
plt.plot(epochs, history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, history.history['acc'], label='acc')
plt.plot(epochs, history.history['val_acc'], label='val_acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.close()

score = model.evaluate(xTest, yTest, batch_size=128, verbose=0)
print('loss : ', score[0], 'acc : ', score[1])

# 모델 외부파일로 저장
model.save('tf27model.keras')
del model

# 저장한 모델 다시 불러오기
myModel = tf.keras.models.load_model('tf27model.keras')
print(xTest[:1], xTest[:1].shape)
plt.imshow(xTest[:1].reshape(28, 28), cmap='Greys')
plt.show()
plt.close()

# 모델로 숫자 예측 시켜보기
pred = myModel.predict(xTest[:1])
print('pred : ', pred)
print('예측값 : ', np.argmax(pred, 1))
print('실제값 : ', yTest[:1])
print('실제값 : ', np.argmax(yTest[:1], 1))

# 모델로 내가 만든 숫자 이미지 예측 시켜보기
from PIL import Image

myImgFromOut = Image.open('su.png')
myImg = np.array(myImgFromOut.resize((28, 28), Image.Resampling.LANCZOS).convert('L'))
print(myImg.shape)

data = myImg.reshape([1, 784]).astype('float32')
data = data / 255.0

myPred = myModel.predict(data)
print('pred : ', myPred)
print('예측값 : ', np.argmax(myPred, 1))
plt.imshow(data.reshape(28 , 28), cmap = 'Greys')
plt.show()
plt.close()
