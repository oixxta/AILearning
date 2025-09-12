"""
캘리포니아 주택 가격 데이터로 함수형 유연한 모델 생성 : Functional API 사용

"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
import matplotlib.pylab as plt

# 데이터 가져오기
housing = fetch_california_housing()
print(housing.keys())
print(housing.data[:3], type(housing.data))
print(housing.target[:3], type(housing.target)) 
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(housing.feature_names)    
#['MedHouseVal']
print(housing.target_names)
print(housing.data.shape)       #(20640, 8)

# 전체데이터를 train/test 데이터 분리
xTrain_all, xTest, yTrain_all, yTest = train_test_split(housing.data, housing.target, test_size=0.2, random_state=12)
print(xTrain_all.shape, xTest.shape, yTrain_all.shape, yTest.shape) #(16512, 8) (4128, 8) (16512,) (4128,)

# Train데이터를 train/validation 데이터 분리
xTrain, xValid, yTrain, yValid = train_test_split(xTrain_all, yTrain_all, test_size=0.3, random_state=12)
print(xTrain.shape, xValid.shape, yTrain.shape, yValid.shape) #(11558, 8) (4954, 8) (11558,) (4954,)

print(xTrain[:3])
"""
[[ 5.63580000e+00  2.90000000e+01  6.28376844e+00  9.78433598e-01
   2.65100000e+03  3.00908059e+00  3.72700000e+01 -1.21920000e+02]
 [ 4.69370000e+00  2.60000000e+01  5.48401163e+00  1.00872093e+00
   2.10300000e+03  3.05668605e+00  3.41300000e+01 -1.17840000e+02]
 [ 3.74540000e+00  5.20000000e+01  4.60377358e+00  1.00754717e+00
   4.88000000e+02  1.84150943e+00  3.66200000e+01 -1.21910000e+02]]
"""

# 표준화 하기
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)
xValid = scaler.fit_transform(xValid)
print(xTrain[:3])
"""
[[ 9.44857245e-01  2.19318421e-02  3.09807859e-01 -2.20018179e-01
   1.10664684e+00 -7.52009820e-03  7.70693427e-01 -1.17771761e+00]
 [ 4.41541326e-01 -2.17509182e-01  1.76786729e-02 -1.64369194e-01
   6.12244680e-01 -1.29091018e-03 -6.99544393e-01  8.60574608e-01]
 [-6.50869353e-02  1.85764636e+00 -3.03848093e-01 -1.66525824e-01
  -8.44798188e-01 -1.60297129e-01  4.66344834e-01 -1.17272180e+00]]
"""


# Sequential API 사용해보기 : 단순한 방법으로 다층퍼셉트론(MLP)
def modelSequential():
    model1 = Sequential()
    model1.add(Input(shape=xTrain.shape[1:]))
    model1.add(Dense(units=32, activation='relu'))
    model1.add(Dense(units=1, activation='linear'))
    print(model1.summary())
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    dense (Dense)               (None, 32)                288

    dense_1 (Dense)             (None, 1)                 33

    =================================================================
    Total params: 321
    Trainable params: 321
    Non-trainable params: 0
    _________________________________________________________________
    """
    model1.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history1 = model1.fit(xTrain, yTrain, epochs=20, validation_data=(xValid, yValid), verbose=2)

    print('evaluate : ', model1.evaluate(xTest, yTest, verbose=0))  #[3.268509864807129, 3.268509864807129]

    # test 일부 자료로 예측값 확인해보기
    xNew = xTest[:3]
    yPred = model1.predict(xNew)
    print('예측값 : ', yPred.ravel())    #[1.4377048  4.665251   1.9943975]
    print('실제값 : ', yTest[:3])        #[2.114,     1.952,     2.418]

    # 시각화로 확인
    plt.plot(range(1, 21), history1.history['mse'], c='b', label='mse')
    plt.plot(range(1, 21), history1.history['val_mse'], c='r', label='val_mse')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()
    plt.show()
    plt.close()


# Fucntional API 사용해보기 : 유연한 모델, 모든 특성 전부 깊은경로로 전달
def modelFucntional1():
    input_ = Input(shape=xTrain.shape[1:])
    net1 = Dense(units=32, activation='relu')(input_)
    net2 = Dense(units=32, activation='relu')(net1)
    concat = Concatenate()([input_, net2])     #입력층과 마지막 출력층을 묶어서 하나의 층 만들기
    output = Dense(units=1)(concat)

    model2 = Model(inputs=[input_], outputs=[output])

    model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history2 = model2.fit(xTrain, yTrain, epochs=20, validation_data=(xValid, yValid), verbose=2)

    print('evaluate : ', model2.evaluate(xTest, yTest, verbose=0))  #evaluate :  [6.881189346313477, 6.881189346313477]

    # test 일부 자료로 예측값 확인해보기
    xNew = xTest[:3]
    yPred = model2.predict(xNew)
    print('예측값 : ', yPred.ravel())    #예측값 :  [1.6439451 6.097655  2.058981 ]
    print('실제값 : ', yTest[:3])        #실제값 :  [2.114 1.952 2.418]

    # 시각화로 확인
    plt.plot(range(1, 21), history2.history['mse'], c='b', label='mse')
    plt.plot(range(1, 21), history2.history['val_mse'], c='r', label='val_mse')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()
    plt.show()
    plt.close()


# Fucntional API 사용해보기 : 일부 특성은 짧은 경로로 전달, 나머지는 깊은 경로로 전달.
# 예를 들어, 8개의 feature 중 5개(0 ~ 4)는 짧은 경로로, 6개의 특성(2 ~ 7)은 깊은 경로로 전달
def modelFucntional2():
    inputA = Input(shape=[5], name='wide_input')
    inputB = Input(shape=[6], name='deep_input')
    net1 = Dense(units=32, activation='relu')(inputB)
    net2 = Dense(units=32, activation='relu')(net1)
    concat = Concatenate()([inputA, net2])
    output = Dense(units=1, activation='linear', name='output')(concat)

    model3 = Model(inputs=[inputA, inputB], outputs=[output])

    model3.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # fit을 호출할 때 하나의 입력행렬 xTrain을 전달하는 것이 아니라,
    # 입력마다 하나씩 행렬의 튜플(xTrain_a, xTrain_b)을 전달해야 함.
    xTrain_a, xTrain_b = xTrain[:, :5], xTrain[:, 2:8]
    xValid_a, xValid_b = xValid[:, :5], xValid[:, 2:8]
    xTest_a, xTest_b = xTest[:, :5], xTest[:, 2:8]       # evaluate용


    history3 = model3.fit((xTrain_a, xTrain_b), yTrain, epochs=20, validation_data=((xValid_a, xValid_b), yValid), verbose=2)

    print('evaluate : ', model3.evaluate((xTest_a, xTest_b), yTest, verbose=0))  #evaluate :  [6.881189346313477, 6.881189346313477]

    # test 일부 자료로 예측값 확인해보기
    xNew_a, xNew_b = xTest_a[:3], xTest_b[:3]
    yPred = model3.predict((xNew_a, xNew_b))
    print('예측값 : ', yPred.ravel())    #예측값 :  [1.6439451 6.097655  2.058981 ]

    print('실제값 : ', yTest[:3])        #실제값 :  [2.114 1.952 2.418]

    # 시각화로 확인
    plt.plot(range(1, 21), history3.history['mse'], c='b', label='mse')
    plt.plot(range(1, 21), history3.history['val_mse'], c='r', label='val_mse')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()
    plt.show()
    plt.close()


# Model Subclassing API 사용해보기 : 모든 특성 전부 깊은경로로 전달
def modelSubclassing1():
    pass

modelSequential()
modelFucntional1()
modelFucntional2()