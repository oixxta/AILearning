"""
케라스 모듈로 논리회로 분류 모델 작성해보기

"""
import numpy as np
import tensorflow as tf
#from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

def func1():
    print("TF version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))


def func2():
    #데이터 세트 생성
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])

    #모델 구성
    model = Sequential([
        Input(shape=(2,)),
        Dense(units=1),
        Activation('sigmoid'),
    ])

    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))

    #모델 학습 과정
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


func1()



# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"