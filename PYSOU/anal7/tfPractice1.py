"""
문제 1)

https://github.com/data-8/materials-fa17/blob/master/lec/galton.csv

data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
 - train / test 분리
 - Sequential api와 function api 를 사용해 모델을 만들어 보시오.
 - train과 test의 mse를 시각화 하시오
 - 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#데이터 읽어오기
originData = pd.read_csv("https://raw.githubusercontent.com/data-8/materials-fa17/refs/heads/master/lec/galton.csv")
print(originData.head(3))
"""
  family  father  mother  midparentHeight  children  childNum  gender  childHeight
0      1    78.5    67.0            75.43         4         1    male         73.2
1      1    78.5    67.0            75.43         4         2  female         69.2
2      1    78.5    67.0            75.43         4         3  female         69.0
"""
print(originData.columns)   #['family', 'father', 'mother', 'midparentHeight', 'children', 'childNum', 'gender', 'childHeight']
print(originData.shape)     #(934, 8)

#데이터 전처리
#1. 전체 데이터에서 gender가 male인 것들만 사용
data = originData[originData['gender'] == 'male']
print(data.head(3))
print(data.shape)       #(481, 8)
#2. 필요한 칼럼만 이용 : father, childHeight
data = data[['father', 'childHeight']]
print(data.head(3))
print(data.shape)       #(481, 2)
#3. 피쳐와 레이블로 나누기
xData = np.array(data['father'], dtype=np.float32).reshape(-1, 1)
yData = np.array(data['childHeight'], dtype=np.float32).reshape(-1, 1)
#4. 데이터를 테스트용과 학습용으로 나누기
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=1, stratify=True)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(336, 1) (145, 1) (336, 1) (145, 1)


#모델 만들기1 : Sequential API 사용
def model1():
    model = Sequential()       #계층 구조 (리니어 레이어 스택)
    model.add(Input(1,))       #입력층 (1개)
    model.add(Dense(units=8, activation='relu'))   #히든레이어 1
    model.add(Dense(units=1, activation='linear'))  #출력층 (1개)

    print(model.summary())
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
    opti = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opti, loss='mse', metrics=['mse'])

    history = model.fit(x=xTrain, y=yTrain, batch_size=16, epochs=100, verbose=0)
    lossMetrics = model.evaluate(x=xTrain, y=yTrain, verbose=0)
    print('loss_metrics : ', lossMetrics)

    #모델 성능 확인하기
    yPred = model.predict(xTest, verbose=0)
    print('실제값 : ', yTest.ravel()[:5])   #실제값 :  [68.5      69.       71.       68.       70.      ]
    print('예측값 : ', yPred.ravel()[:5])   #예측값 :  [66.55309  66.97858  69.5315   65.27663  64.851135]
    print('설명력 : ', r2_score(yTest, yPred))  #설명력 :  -0.6181730031967163

    #시각화 해보기
    plt.plot(xTest.ravel(), yPred.ravel(), 'b', label='pred')
    plt.plot(xTest.ravel(), yTest.ravel(), 'ko', label='real')
    plt.legend()
    plt.show()
    plt.close()

    #새로운 아버지의 키 데이터로 아들의 키가 어느정도일지 예측해보기
    newData(model)


#모델 만들기2 : function API 사용
def model2():
    input = Input(shape=(1,))   #입력층 (입력 1개)
    hiddenLayer = Dense(units=8, activation='relu')(input) #히든레이어
    output = Dense(units=1, activation='linear')(hiddenLayer)   #출력층 (출력 1개)
    model = Model(input, output)     #모델 생성 및 입력값과 출력값 지정

    print(model.summary())
    """
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)        [(None, 1)]               0

    dense_2 (Dense)             (None, 8)                 16        

    dense_3 (Dense)             (None, 1)                 9

    =================================================================
    Total params: 25
    Trainable params: 25
    Non-trainable params: 0
    _________________________________________________________________
    """
    opti = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opti, loss='mse', metrics=['mse'])

    history = model.fit(x=xTrain, y=yTrain, batch_size=16, epochs=100, verbose=0)
    lossMetrics = model.evaluate(x=xTrain, y=yTrain, verbose=0)
    print('loss_metrics : ', lossMetrics)
    print(history)

    #모델 성능 확인하기
    yPred = model.predict(xTest, verbose=0)
    print('실제값 : ', yTest.ravel()[:5])   #실제값 :  [68.5     69.      71.      68.      70.     ]
    print('예측값 : ', yPred.ravel()[:5])   #예측값 :  [68.58675 69.08012 72.04036 67.10663 66.61326]
    print('설명력 : ', r2_score(yTest, yPred))  #설명력 : -0.09230029582977295

    #시각화 해보기
    plt.plot(xTest.ravel(), yPred.ravel(), 'b', label='pred')
    plt.plot(xTest.ravel(), yTest.ravel(), 'ko', label='real')
    plt.legend()
    plt.show()
    plt.close()

    #새로운 아버지의 키 데이터로 아들의 키가 어느정도일지 예측해보기
    newData(model)


#키 예측
def newData(model):
    newData = input('새로운 키 데이터 입력 : ')
    print('입력받은 키 : ', newData)
    try:
        newVal = float(newData)
        newInput = np.array([[newVal]], dtype=np.float32)  # 모델 입력 형태로 reshape
        newPred = model.predict(newInput, verbose=0)
        print('예측된 아들의 키 : {:.2f}'.format(newPred.ravel()[0]))
    except ValueError:
        print("숫자를 정확히 입력하세요 ")


model1()
model2()