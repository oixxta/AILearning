"""
텐서플로우의 세 가지 종류의 신경 구조 설계 사용해보기! : Sequential, Functional, Model Subclassing

단순 선형회귀 모델로 생성.
"""
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#공부시간에 따른 성적 데이터 사용
xData = np.array([1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
#xData = [[1], [2], [3], [4] ,[5]]와 같음.
yData = np.array([11, 32, 53, 64, 70], dtype=np.float32).reshape(-1, 1)


#모델 생성방법 1 : Sequential API 사용
#특징 : 모델 구성이 순차적이거나 단순한 경우에 사용함
def model1():
    model1 = Sequential()       #계층 구조 (linear layer stack)
    model1.add(Input(1,))
    model1.add(Dense(units=16, activation='relu'))
    model1.add(Dense(units=1, activation='linear'))

    print(model1.summary())
    """
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    dense (Dense)               (None, 16)                32

    dense_1 (Dense)             (None, 1)                 17

    =================================================================
    Total params: 49
    Trainable params: 49
    Non-trainable params: 0
    _________________________________________________________________
    """
    opti = optimizers.SGD(learning_rate=0.001)
    model1.compile(optimizer=opti, loss='mse', metrics=['mse'])      #MSE : 수치가 0에 가까울수록 좋음.

    history = model1.fit(x=xData, y=yData, batch_size=1, epochs=100, verbose=2)
    loss_metrics = model1.evaluate(x=xData, y=yData, verbose=0)
    print('loss_metrics : ', loss_metrics)  #배치 사이즈는 1, 데이터는 5개임으로 5/5로 표현됨.

    #성능확인
    yPred = model1.predict(xData, verbose=0)
    print('실제값 : ', yData.ravel())           #실제값 :  [11.       32.       53.       64.       70.      ]
    print('예측값 : ', yPred.ravel())           #예측값 :  [17.500502 32.289303 47.07811  61.86691  76.65571 ]
    print('설명력 : ', r2_score(yData, yPred))  #설명력 :  0.9467267990112305

    #새 데이터(n, 1)로 예측해보기
    newData = np.array([1.5, 2.2, 5.8], dtype=np.float32).reshape(-1, 1)
    newPred = model1.predict(newData, verbose=0).ravel()
    print('새 예상점수 : ', newPred)            #[25.131096 35.56215  89.29083 ]

    #시각화 해보기
    plt.plot(xData.ravel(), yPred.ravel(), 'b', label='pred')
    plt.plot(xData.ravel(), yData.ravel(), 'ko', label='real')
    plt.legend()
    plt.show()
    plt.close()


#모델 생성방법 2 : Functional API 사용
#특징 : Sequential과 비교해서 더 유연한 신경구조를 가짐. 입력데이터로 여러 층을 공유, 다양한 종류의 입출력 가능.
#mulit input modle, multi output model, 공유층 활용 모델, 비순차적 데이터 처리 등이 가능함. 
def model2():
    #히든레이어가 없을때
    #inputs = Input(shape=(1,))                                  #입력크기 지정
    #outputs = Dense(units=1, activation='linear')(inputs)       #출력크기 지정
    #model2 = Model(inputs, outputs)     #입력값, 출력값 지정

    #히든레이어가 있을때 : 이전 층을 다음 층 함수의 입력으로 사용하기 위해 변수 할당
    inputs = Input(shape=(1,))                                  #입력크기 지정
    output1 = Dense(units=16, activation='relu')(inputs)        #히든레이어
    output2 = Dense(units=1, activation='linear')(output1)      #출력
    model2 = Model(inputs, output2)     #입력값, 출력값 지정, 완성된 모델.
    #model2 = Model([입력1, 입력2], [출력1, 출력2, 출력3])  #입력값과 출력값이 여러개일때.

    opti2 = optimizers.SGD(learning_rate=0.001)
    model2.compile(optimizer=opti2, loss='mse', metrics=['mse'])

    history2 = model2.fit(x=xData, y=yData, batch_size=1, epochs=100, verbose=2)
    loss_metrics = model2.evaluate(x=xData, y=yData, verbose=0)
    print('loss_metrics : ', loss_metrics)

    #성능확인
    yPred = model2.predict(xData, verbose=0)
    print('실제값 : ', yData.ravel())           
    print('예측값 : ', yPred.ravel())           
    print('설명력 : ', r2_score(yData, yPred))



#모델 생성방법 3 : Model Subclassing API 사용
#특징 : 고난이도 작업에서 활용성 높음, 정적이지 않고 동적인 구조에 적합함.
def model3():
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.d1 = Dense(16, activation='relu')      #히든레이어
            self.d2 = Dense(1, activation='linear')     #출력층

        # x는 input 매개변수 : 함수형 API와 유사하나, Input 객체를 생성하진 않음.
        # 계산 작업 등을 할 수 있다.
        # model.fit(), evaluate(), predict() 등을 하면 자동으로 호출됨.
        def call(self, x):  
            x = self.d1(x)      #input 객체를 만들어서 히든레이어에 넣음.
            return self.d2(x)   #출력층의 결과를 반환함.
    
    model3 = MyModel()

    opti3 = optimizers.SGD(learning_rate=0.001)
    model3.compile(optimizer=opti3, loss='mse', metrics=['mse'])

    history2 = model3.fit(x=xData, y=yData, batch_size=1, epochs=100, verbose=2)
    loss_metrics = model3.evaluate(x=xData, y=yData, verbose=0)
    print('loss_metrics : ', loss_metrics)

    #성능확인
    yPred = model3.predict(xData, verbose=0)
    print('실제값 : ', yData.ravel())           
    print('예측값 : ', yPred.ravel())           
    print('설명력 : ', r2_score(yData, yPred))


#Model Subclassing API 사용 : 모델 뿐만 아니라 레이어도 상속받는 방식으로 만들기
def model3_1():
    from keras.layers import Layer

    # Layer (사용자 정의층 작성용)
    # 케라스의 layers 패키지에 정의된 레이어 대신 새로운 연산을 하는 레이어 혹은 편의를
    # 위해 여러 레이어를 하나로 묶은 레이어를 구현할 때 사용 가능.
    class MyLinear(Layer):
        def __init__(self, units=1, **kwargs):
            super(MyLinear, self).__init__(**kwargs)    #하이퍼 파라미터 생성
            # 여기에 여러 레이어를 하나로 묶은 레이어를 구현할 수 있음.
            # ...
            self.units = units
        
        # 빌드는 내부적으로 자동으로 call을 호출함. 모델의 가중치 관련 내용을 기술함.
        def build(self, input_shape):
            print('build : input_shape = {}'.format(input_shape))
            self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True) #역전파를 진행할 때 w값을 갱신해야함.
            self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

        # 정의된 값들을 이용해 해당 층의 로직을 정의함.
        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b
    
    class MyMlp(Model):
        def __init__(self, **kwargs):
            super(MyMlp, self).__init__(**kwargs)
            self.linear1 = MyLinear(2)
            self.linear2 = MyLinear(1)

        def call(self, inputs):
            x = self.linear1(inputs)
            x = tf.nn.relu(x)
            return self.linear2(x)
        
    model4 = MyMlp()

    model4.summary()

    opti4 = optimizers.SGD(learning_rate=0.001)
    model4.compile(optimizer=opti4, loss='mse', metrics=['mse'])

    history3 = model4.fit(x=xData, y=yData, batch_size=1, epochs=100, verbose=2)
    loss_metrics = model4.evaluate(x=xData, y=yData, verbose=0)
    print('loss_metrics : ', loss_metrics)

    #성능확인
    yPred = model4.predict(xData, verbose=0)
    print('실제값 : ', yData.ravel())           
    print('예측값 : ', yPred.ravel())           
    print('설명력 : ', r2_score(yData, yPred))


#model1()
#model2()
#model3()
model3_1()