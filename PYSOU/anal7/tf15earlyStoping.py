"""
자동차 연비 예측 : 
다중선형회귀, Sequential API 사용, Network 구성 함수 작성, 조기종료(Early Stoping) 사용
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf


#데이터 가져오기
url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/auto-mpg.csv"
dataset = pd.read_csv(url, na_values='?')


#데이터 가공
del dataset['car name']
print(dataset.head(3))
"""
    mpg  cylinders  displacement  horsepower  weight  acceleration  model year  origin
0  18.0          8         307.0       130.0    3504          12.0          70       1
1  15.0          8         350.0       165.0    3693          11.5          70       1
2  18.0          8         318.0       150.0    3436          11.0          70       1
"""
print(dataset.columns)  
#['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
dataset.drop(['cylinders', 'acceleration', 'model year', 'origin'], axis='columns', inplace=True)
print(dataset.shape)    #(398, 4)
print(dataset.corr())   #상관관계 확인
"""
                   mpg  displacement  horsepower    weight
mpg           1.000000     -0.804203   -0.778427 -0.831741
displacement -0.804203      1.000000    0.897257  0.932824
horsepower   -0.778427      0.897257    1.000000  0.864538
weight       -0.831741      0.932824    0.864538  1.000000
"""
print(dataset.info())
"""
RangeIndex: 398 entries, 0 to 397
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   mpg           398 non-null    float64
 1   displacement  398 non-null    float64
 2   horsepower    392 non-null    float64
 3   weight        398 non-null    int64
dtypes: float64(3), int64(1)
마력에 결측치 존재
"""
print(dataset.isna().sum())     #horsepower에 결측치 6개
dataset = dataset.dropna()      #결측치 제거

#상관관계 시각화로 확인
sns.pairplot(dataset[['mpg', 'displacement', 'horsepower', 'weight']], diag_kind='kde')
plt.show()
plt.close()


# 연습데이터와 훈련데이터 분리
trainDataset = dataset.sample(frac=0.7, random_state=123)
testDataset = dataset.drop(trainDataset.index)
print(trainDataset.shape, testDataset.shape)        #(274, 4) (118, 4)


# 표준화 - 함수가 아니라 수식으로 (관찰값 - 평균) / 표준편차
trainStat = trainDataset.describe()
print(trainStat)
trainStat.pop('mpg')
print(trainStat.transpose())
"""
              count         mean         std     min      25%     50%      75%     max
displacement  274.0   196.131387  106.618440    70.0   101.75   151.0   302.00   455.0
horsepower    274.0   104.755474   39.416747    46.0    75.00    94.0   129.00   230.0
weight        274.0  2981.941606  863.904789  1613.0  2190.00  2831.5  3641.75  4997.0
"""
trainStat = trainStat.transpose()

def std_func(x):
    return (x - trainStat['mean']) / trainStat['std']

stTrainData = std_func(trainDataset)
stTrainData = stTrainData.drop(['mpg'], axis='columns')
print(stTrainData[:2])
"""
     displacement  horsepower    weight
222      0.599039    0.133053  1.247890
247     -1.042328   -0.881744 -1.055604
"""
stTestData = std_func(testDataset)
stTestData = stTestData.drop(['mpg'], axis='columns')
print(stTestData[:2])
"""
   displacement  horsepower    weight
1      1.443171    1.528399  0.823075
2      1.143035    1.147850  0.525588
"""
trainLabel = trainDataset.pop('mpg')
print(trainLabel[:2])
testLabel = testDataset.pop('mpg')
print(testLabel[:2])


#모델 만들기
def build_model():
    network = Sequential([
        Input(shape=3,),
        Dense(units=32, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='linear')
    ])

    opti = tf.keras.optimizers.Adam(learning_rate=0.01)
    network.compile(optimizer=opti, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])
    return network

model = build_model()
print(model.summary())


#모델 학습시키기
epoch = 5000
earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', baseline=0.03, patience=5)    #얼리스탑 객체변수
#조기종료 : 학습이 끝나면 에포크가 남아도 학습을 종료함.
#validation loss를 사용함.
#baseline : loss값이 0.03에 도달하면 자동 종료, 실무에선 5 ~ 7정도를 사용함.
#patience : 0.03에 도달하지 않아도 0.03 근처에 5회 도달하면 학습을 멈춤.

history = model.fit(stTrainData, trainLabel, batch_size=32, epochs=epoch, validation_split=0.2, verbose=2, callbacks=[earlyStop])
df = pd.DataFrame(history.history)
print(df.head(5))       #5000번을 다 안하고 5번만에 끝남.


#모델 학습 정보 시각화 하기
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    #print(hist.head(3))
    plt.figure(figsize=(8, 12))

    plt.subplot(2, 1, 1)
    plt.xlabel('epoch')
    plt.ylabel('mean squared error [mpg]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='train error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='validation error')

    plt.subplot(2, 1, 2)
    plt.xlabel('epoch')
    plt.ylabel('mean_absolute_error [$mpg^2$]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='train abs error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='validation abs error')

    plt.legend()
    plt.show()

plot_history(history)

loss, mse, mae = model.evaluate(stTestData, testLabel)
print('test dataset으로 평가하기 : loss값 : {:5.3f}'.format(loss))  # 69.683
print('test dataset으로 평가하기 : mse값 : {:5.3f}'.format(mse))    # 69.683
print('test dataset으로 평가하기 : mae값 : {:5.3f}'.format(mae))    # 6.571


#새로운 값 예측하기
#'displacement', 'horsepower', 'weight'
newData = pd.DataFrame({'displacement' : [300, 400], 'horsepower' : [120, 140], 'weight' : [2000, 4000]})
stNewData = std_func(newData)   #표준화 실시
newPred = model.predict(stNewData).ravel()
print('예측 결과 : ', newPred)  #예측 결과 :  [ 8.933093 12.646007]