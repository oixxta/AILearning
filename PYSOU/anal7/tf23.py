"""
Sigmoid는 softmax로 처리 가능
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

dataset = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/diabetes.csv', delimiter=',')
print(dataset.shape)
print(dataset[:1])
print(set(dataset[:, -1]))

# 이항분류(시그모이드)
xTrain, xTest, yTrain, yTest = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, shuffle=True, random_state=123)
print(xTrain.shape, xTest.shape)  #(531, 8) (228, 8)

model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(xTrain, yTrain, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
scores = model.evaluate(xTest, yTest)
print('%s : %.2f%%'%(model.metrics_names[1], scores[1] * 100))


# 다항분류(소프트맥스)
xTrain, xTest, yTrain, yTest = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, shuffle=True, random_state=123)
print(xTrain.shape, xTest.shape)

# 레이블에 원 핫 인코딩 필요!
yTrain = to_categorical(yTrain)
yTest = to_categorical(yTest)

model2 = Sequential()
model2.add(Input(shape=(8,)))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(2, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model2.summary())

model2.fit(xTrain, yTrain, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
scores = model2.evaluate(xTest, yTest)
print('%s : %.2f%%'%(model2.metrics_names[1], scores[1] * 100))