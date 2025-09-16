"""
딥 러닝을 활용한 이항분류 연습 : Sequantial, Functional, SubclassModeling
"""
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Functional, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

#데이터 만들기
xData = np.array([[1, 2], [2, 3], [3, 4], [4, 3], [3, 2], [2, 1]], dtype=np.float32)  #앞의 3개는 증강, 뒤의 3개는 감소하는 패턴
yData = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

#모델 정의 방법 1 : Sequential
def modelOne():
    #modelSequential = Sequential([
    #    Input(shape=(2,)),
    #    Dense(units=1, activation='sigmoid')
    #])
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  #분류는 loss를 엔트로피를 씀. 이 경우는 이항분류이니 binary_crossentropy
    model.summary()
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    dense (Dense)               (None, 1)                 3

    =================================================================
    Total params: 3
    Trainable params: 3
    Non-trainable params: 0
    _________________________________________________________________
    """
    model.fit(xData, yData, epochs=50, batch_size=1, verbose=2)
    m_eval = model.evaluate(xData, yData, verbose=0)
    print(f'평가 결과 : (손실 loss, ) : {m_eval[0]}, (정확도, accuracy) : {m_eval[1]:.4f}')

    #예측해보기
    newData = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
    pred = model.predict(newData, verbose=0)
    print('예측 확률값 : ', pred.ravel())
    print('예측 결과 : ', [1 if i >= 0.5 else 0 for i in pred])
    print('예측 결과 : ', (pred >= 0.5).astype(int).ravel())
    print('예측 결과  : ', np.where(pred >= 0.5, 1, 0).ravel())
#모델 정의 방법 2 : Functional
def modelTwo():
    inputLayer = Input(shape=(2,))
    outputLayer = Dense(1, activation='sigmoid')(inputLayer)

    model = Model(inputs = inputLayer, outputs = outputLayer)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  #분류는 loss를 엔트로피를 씀. 이 경우는 이항분류이니 binary_crossentropy
    model.summary()
    model.fit(xData, yData, epochs=50, batch_size=1, verbose=2)
    m_eval = model.evaluate(xData, yData, verbose=0)
    print(f'평가 결과 : (손실 loss, ) : {m_eval[0]}, (정확도, accuracy) : {m_eval[1]:.4f}')

    #예측해보기
    newData = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
    pred = model.predict(newData, verbose=0)
    print('예측 확률값 : ', pred.ravel())
    print('예측 결과 : ', [1 if i >= 0.5 else 0 for i in pred])
    print('예측 결과 : ', (pred >= 0.5).astype(int).ravel())
    print('예측 결과  : ', np.where(pred >= 0.5, 1, 0).ravel())

#모델 정의 방법 3 : modelSubclassing
class Mymodel(Model):
    def __init__(self):
        super().__init__(name='MyBinaryClass')
        self.dense = Dense(1, activation='sigmoid', name='dense_sigmoid')

    def build(self, input_shape):
        #첫 번째 순방향(feed forward)에서 가중치를 만든다.
        super().build(input_shape)

    def call(self, inputs, training=False):
        print('>>> call 실행됨, trainging : ', training)
        return self.dense(inputs)
    

model = Mymodel()
model.build(input_shape=(None, 2))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  #분류는 loss를 엔트로피를 씀. 이 경우는 이항분류이니 binary_crossentropy
model.summary()
model.fit(xData, yData, epochs=50, batch_size=1, verbose=2)
m_eval = model.evaluate(xData, yData, verbose=0)
print(f'평가 결과 : (손실 loss, ) : {m_eval[0]}, (정확도, accuracy) : {m_eval[1]:.4f}')

#예측해보기
newData = np.array([[1, 2.5], [10.5, 7.1]], dtype=np.float32)
pred = model.predict(newData, verbose=0)
print('예측 확률값 : ', pred.ravel())
print('예측 결과 : ', [1 if i >= 0.5 else 0 for i in pred])
print('예측 결과 : ', (pred >= 0.5).astype(int).ravel())
print('예측 결과  : ', np.where(pred >= 0.5, 1, 0).ravel())