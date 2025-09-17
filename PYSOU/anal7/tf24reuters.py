"""
로이터 뉴스 데이터로 다항분류 해보기
"""
from keras.datasets import reuters
print(reuters.load_data(num_words=10000))

(trainData, trainLabel), (testData, testLabel) = reuters.load_data(num_words=10000)
print(len(trainData), len(testData))    #8982 2246
print(trainData[0])
print(trainLabel[0])
print(set(trainLabel))

#실제 데이터 읽기
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#print(reverse_word_index)
decord_review = ' '.join([reverse_word_index.get(i) for i in trainData[0]])
print(decord_review)


import numpy as np
#데이터 준비
def vector_seq(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))    #0으로 가득 찬 2차원 배열 생성
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

xTrain = vector_seq(trainData)
xTest = vector_seq(testData)
print(xTest)
import sys
np.set_printoptions(threshold=sys.maxsize)
print(xTest)

#one hot 인코딩 하기
"""
def to_onehot(labels, dim=46):
    results = np.zeros((len(labels), dim))    #0으로 가득 찬 2차원 배열 생성
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

one_hot_train_labels = to_onehot(trainLabel)
one_hot_test_labels = to_onehot(testLabel)
print(one_hot_test_labels[0])
"""
from keras.utils import to_categorical
one_hot_train_labels = to_categorical(trainLabel)
one_hot_test_labels = to_categorical(testLabel)
print(one_hot_test_labels[0])


#신경망 모델 만들기
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import models

model = models.Sequential()
model.add(Input(shape=(10000,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())


#벨리데이션 데이터 만들기 (벨리데이션 스플릿 안씀)
xVal = xTrain[:1000]
partialXTrain = xTrain[1000:]
yVal = one_hot_train_labels[:1000]
partialYTrain = one_hot_train_labels[1000:]

#모델 학습하기
history = model.fit(partialXTrain, partialYTrain, epochs=50, batch_size=128, validation_data=(xVal, yVal), verbose=2)

#모델 평가 및 히스토리 시각화
result = model.evaluate(xTest, one_hot_test_labels)
print(result)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='validation acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.close()