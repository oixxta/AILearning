"""
문제1)
https://www.kaggle.com/jyotikumarrout/graduation 의 binary.csv 데이터를 이용하여 
미국 대학원 입학여부를 분류하는 모델을 작성하시오. loss, accuracy에 대한 시각화도 실시한다.
input 함수를 사용해 새로운 gre, gpa, rank 값을 받아  admit을 판정하시오.
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras import optimizers
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def classificationOne():
    #데이터 가져오기
    data = pd.read_csv('binary.csv')
    print(data.head(3))
    """
             admit  gre   gpa  rank
        0      0    380  3.61     3
        1      1    660  3.67     3
        2      1    800  4.00     1
    """
    print(data.columns)         #Index(['admit', 'gre', 'gpa', 'rank'], dtype='object')
    print(data.isna().sum())    #결측치 없음
    print(data.shape)           #(400, 4)
    print(len(data[data.iloc[:, 0] == 0]))  #불합격 : 273
    print(len(data[data.iloc[:, 0] == 1]))  #합격 : 127

    #데이터 라벨이랑 피쳐 나누기
    xData = data.drop(columns=['admit'])
    yData = data['admit']

    #학습데이터 나누기
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=0, shuffle=True, stratify=yData)
    print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
    print(xTest[:3])
    """
         gre   gpa  rank
    307  580  3.51     2
    237  480  4.00     2
    142  620  3.94     4
    """
    print(yTest[:3])
    """
    307    0
    237    0
    142    0
    """

    # 3) 전처리 파이프라인: 수치형 표준화 + rank 원핫(1~4, 첫 범주 드롭)
    ss = StandardScaler()
    xTrain = ss.fit_transform(xTrain)
    xTest  = ss.transform(xTest)
    
    #모델 생성하기 : Sequential
    model = Sequential()
    model.add(Input(shape=(3,)))
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    loss, acc = model.evaluate(xTrain, yTrain, verbose=0)    #fit을 하기 전에 모델의 score를 확인할 수 있음.
    print('훈련 전 모델 정확도 : {:5.2f}%'.format(100 * acc))   #훈련 전 모델 정확도 : 31.79%

    #모델 학습하기 : 얼리스탑도 사용
    earlyStop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(xTrain, yTrain, validation_split=0.2, epochs=1000, batch_size=64, callbacks=[earlyStop], verbose=2)
    loss, acc = model.evaluate(xTest, yTest, batch_size=64, verbose=0)
    print('훈련 후 모델 정확도 : {:5.2f}%'.format(100 * acc))   #훈련 후 모델 정확도 : 72.50%

    #시각화 해보기
    epochLen = np.arange(len(history.epoch))
    plt.plot(epochLen, history.history['val_loss'], label='val_loss')
    plt.plot(epochLen, history.history['loss'], label='loss', c='red')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.plot(epochLen, history.history['val_accuracy'], label='val_accuracy')
    plt.plot(epochLen, history.history['accuracy'], label='accuracy', c='blue')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    #새로운 gre, gpa, rank 값을 받아 admit을 판정해보기.
    newGre = float(input('새로운 Gre값 : '))
    newGpa = float(input('새로운 Gpa값 : '))
    newRank = float(input('새로운 Rank값 : '))
    newData = np.array([[newGre, newGpa, newRank]], dtype='float32')
    pred = model.predict(newData)
    print('예측 결과 : ', np.where(pred >= 0.5, 1, 0).ravel())
    
classificationOne()


"""
문제2)
21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 당뇨 판정을
위한 분류 모델을 작성한다.

피마 인디언 당뇨병 데이터는 아래와 같이 구성되어 있다.
Pregnancies: 임신 횟수
Glucose: 포도당 부하 검사 수치
BloodPressure: 혈압(mm Hg)
SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값(mm)
Insulin: 혈청 인슐린(mu U/ml)
BMI: 체질량지수(체중(kg)/키(m))^2
DiabetesPedigreeFunction: 당뇨 내력 가중치 값
Age: 나이
Outcome: 5년 이내 당뇨병 발생여부 - 클래스 결정 값(0 또는 1)

당뇨 판정 칼럼은 outcome 이다.   1 이면 당뇨 환자로 판정
train / test 분류 실시
모델 작성은 Sequential API, Function API 두 가지를 사용한다.
loss, accuracy에 대한 시각화도 실시한다.
"""
def classificationTwo():
    


    pass


classificationTwo()