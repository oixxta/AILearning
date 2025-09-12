"""
문제2)

https://github.com/pykwon/python/tree/master/data

자전거 공유 시스템 분석용 데이터 train.csv를 이용하여 대여횟수에 영향을 주는 변수들을 골라 다중선형회귀분석 모델을 작성하시오.

모델 학습시에 발생하는 loss를 시각화하고 설명력을 출력하시오.

새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력하시오.
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터 가져오기 및 전처리
url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv"
originData = pd.read_csv(url)
print(originData.head(5))
"""
              datetime  season  holiday  workingday  weather  temp   atemp  humidity  windspeed  casual  registered  count
0  2011-01-01 00:00:00       1        0           0        1  9.84  14.395        81        0.0       3          13     16
1  2011-01-01 01:00:00       1        0           0        1  9.02  13.635        80        0.0       8          32     40
2  2011-01-01 02:00:00       1        0           0        1  9.02  13.635        80        0.0       5          27     32
3  2011-01-01 03:00:00       1        0           0        1  9.84  14.395        75        0.0       3          10     13
4  2011-01-01 04:00:00       1        0           0        1  9.84  14.395        75        0.0       0           1      1
"""
print(originData.columns)
#['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
originData.drop(['datetime', 'registered', 'casual'], axis='columns', inplace=True) #필요 없는것 드랍
print(originData.shape)           #(10886, 9)
print(originData.isna().sum())    #결측치 없음.

print(originData.corr())  #상관관계 확인; 대여횟수(count)와 상관관계가 제일 높은건 3개 고르기
"""
              season   holiday  workingday   weather      temp     atemp  humidity  windspeed     count
season      1.000000  0.029368   -0.008126  0.008879  0.258689  0.264744  0.190610  -0.147121  0.163439
holiday     0.029368  1.000000   -0.250491 -0.007074  0.000295 -0.005215  0.001929   0.008409 -0.005393
workingday -0.008126 -0.250491    1.000000  0.033772  0.029966  0.024660 -0.010880   0.013373  0.011594
weather     0.008879 -0.007074    0.033772  1.000000 -0.055035 -0.055376  0.406244   0.007261 -0.128655
temp        0.258689  0.000295    0.029966 -0.055035  1.000000  0.984948 -0.064949  -0.017852  0.394454
atemp       0.264744 -0.005215    0.024660 -0.055376  0.984948  1.000000 -0.043536  -0.057473  0.389784
humidity    0.190610  0.001929   -0.010880  0.406244 -0.064949 -0.043536  1.000000  -0.318607 -0.317371
windspeed  -0.147121  0.008409    0.013373  0.007261 -0.017852 -0.057473 -0.318607   1.000000  0.101369
count       0.163439 -0.005393    0.011594 -0.128655  0.394454  0.389784 -0.317371   0.101369  1.000000

temp, atemp, humidity 채택
"""
sns.pairplot(originData[['count', 'temp', 'atemp', 'humidity']], diag_kind='kde')
plt.show()
plt.close()


# 2. 데이터를 연습데이터와 훈련데이터로 분리하기
xData = originData[['temp', 'atemp', 'humidity']]
print(xData.head(3))
yData = originData[['count']]
print(yData.head(3))

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3 ,random_state=0)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(7620, 3) (3266, 3) (7620, 1) (3266, 1)


# 3. 모델 만들기 : 
def mySequentialModel():
    network = Sequential([
        Input(shape=3,),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='linear')
    ])

    opti = tf.keras.optimizers.Adam(learning_rate=0.01)
    network.compile(optimizer=opti, loss='mse', metrics=['mse'])
    return network

model = mySequentialModel()
print(model.summary())
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 16)                64

 dense_1 (Dense)             (None, 1)                 17

=================================================================
Total params: 81
Trainable params: 81
Non-trainable params: 0
"""


# 4. 모델 학습시키기 : 
earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', baseline=0.03, patience=5)
history = model.fit(xTrain, yTrain, batch_size=32, epochs=1000, validation_split=0.2, verbose=2, callbacks=[earlyStop])


# 5. 학습된 모델로 test 데이터 돌려보기 및 모델 설명력 출력 :
yPred = model.predict(xTest)
print('예측값 : ', yPred[:10].ravel())          #[220.14127  203.63243  169.16093  292.35187  208.40953  189.1395 121.267426  40.72195  222.43582  226.2513  ]
print('실제값 : ', yTest[:10].values.ravel())   #[244        239        229        467        335         40      329          2        141        391       ]
r2 = r2_score(yTest.values.ravel(), yPred)     #yTest와 yPred의 차원 1차원으로 꼭 맞추기.
print('r2 score : ', r2)           #0.2071560025215149

# 6. 모델 학습시에 발생하는 loss를 시각화:
hist = pd.DataFrame(history.history)
print(hist.head(3))
"""
           loss           mse      val_loss       val_mse
0  32747.738281  32747.738281  26237.488281  26237.488281
1  26581.816406  26581.816406  25465.548828  25465.548828
2  26329.890625  26329.890625  25329.609375  25329.609375
"""
plt.plot(hist['loss'], label='loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss Value')
plt.show()
plt.close()

# 7. 터미널로 새로운 데이터를 input 함수를 사용해 키보드로 입력하여 대여횟수 예측결과를 콘솔로 출력:
newTemp = float(input('새로운 temp 데이터 입력 : '))
newAtemp = float(input('새로운 atemp 데이터 입력 : '))
newHumidity = float(input('새로운 humidity 데이터 입력 : '))

newData = np.array([[newTemp, newAtemp, newHumidity]], dtype='float32')
newPred = model.predict(newData, verbose=0)
print('새로운 데이터 예측값(count) : ', newPred.ravel())