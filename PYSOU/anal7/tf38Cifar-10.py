"""
CIFAR-10 데이터 세트를 활용해 이미지 분류 모델을 작성해보기

총 10개의 레이블과 60000장의 컬러 이미지 데이터 존재.

이미지 크기 : 32 x 32.

CNN 미사용, CNN 사용 비교해보기
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.datasets import cifar10

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
print('샘플 수 : ', xTrain.shape)
print('채널 수 : ', xTrain.shape[3])
print('이미지 크기 : ', xTrain.shape[1], xTrain.shape[2])
print('test 샘플 수 : ', xTest.shape)
print('test 타입 수 : ', xTest.dtype)

#print(xTrain[0])
#print(yTrain[0])

#시각화
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(xTrain[0], interpolation='bicubic')
plt.subplot(132)
plt.imshow(xTrain[1], interpolation='bicubic')
plt.subplot(133)
plt.imshow(xTrain[2], interpolation='bicubic')
plt.show()


#이미지 정규화 시키기
xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0
#print(xTrain[0])


#레이블에 원 핫 인코딩 처리
NUM_CLASSES = 10
yTrain = to_categorical(yTrain, NUM_CLASSES)
yTest = to_categorical(yTest, NUM_CLASSES)
print(yTrain[0])


#모델 만들기 (Sequential)
"""
model = Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(units=256, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=NUM_CLASSES, activation='softmax'),
])
print(model.summary())
"""
#모델 만들기 (Functional)
inputLayer = Input(shape=(32, 32, 3))
x = Flatten()(inputLayer)
x = Dense(units=256, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
outputLayer = Dense(units=NUM_CLASSES, activation='softmax')(x)
model = Model(inputLayer, outputLayer)
print(model.summary())

#학습하기
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(xTrain, yTrain, batch_size=64, epochs=20, shuffle=True, verbose=2)
print('test acc : ', model.evaluate(xTest, yTest, verbose=0, batch_size=64)[1])
print('test loss : ', model.evaluate(xTest, yTest, verbose=0, batch_size=64)[0])

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

#학습된 모델로 예측해보기
yPred = model.predict(xTest[:10])
print(np.argmax(yPred, axis=-1))
pred_cla = CLASSES[np.argmax(yPred, axis=-1)]
actual_cla = CLASSES[np.argmax(yTest[:10], axis=-1)]
print('예측값 : ', pred_cla)
print('실제값 : ', actual_cla)
print('분류 실패 수 : ', (pred_cla != actual_cla).sum())

#시각화
fig = plt.figure(figsize=(15, 3))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, idx in enumerate(range(len(xTest[:10]))):
    img = xTest[idx]
    ax = fig.add_subplot(1, len(xTest[:10]), i + 1)
    ax.axis('off')
    ax.text(0.5, -0.3, 'pred=' + str(pred_cla[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.8, 'act=' + str(actual_cla[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)
plt.show()