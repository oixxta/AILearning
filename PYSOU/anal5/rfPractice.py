"""
[Randomforest 문제1] 
kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
dataset은 winequality-red.csv 
https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

Input variables (based on physicochemical tests):
 1 - fixed acidity
 2 - volatile acidity
 3 - citric acid
 4 - residual sugar
 5 - chlorides
 6 - free sulfur dioxide
 7 - total sulfur dioxide
 8 - density
 9 - pH
 10 - sulphates
 11 - alcohol
 Output variable (based on sensory data):
 12 - quality (score between 0 and 10)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def RFquestion1():
    data = pd.read_csv("winequality-red.csv")
    print(data.head(3))
    """
        fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
    0            7.4              0.70         0.00             1.9      0.076                 11.0                  34.0   0.9978  3.51       0.56      9.4        5
    1            7.8              0.88         0.00             2.6      0.098                 25.0                  67.0   0.9968  3.20       0.68      9.8        5
    2            7.8              0.76         0.04             2.3      0.092                 15.0                  54.0   0.9970  3.26       0.65      9.8        5
    """
    print(data.columns)
    #['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol', 'quality']
    print(data.shape)   #(1596, 12)
    print(data.info())
    print(data.isnull().any())  #결측치 확인 : 결측치 없음.

    #종속변수 데이터프레임 만들기
    dataY = data['quality']

    #독립변수 데이터프레임 만들기
    dataX = data.drop(columns=['quality'])

    #학습데이터 분리
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2, random_state=1)
    
    # 모델 생성 : 랜덤포레스트
    model = RandomForestClassifier(criterion='entropy', n_estimators=500)
    model.fit(trainX, trainY)

    # 모델 테스트
    pred = model.predict(testX)
    print('예측값 : ', pred[:5])
    print('실제값 : ', testY[:5])
    print('맞춘 갯수 : ', sum(testY == pred) / len(testY))
    print('분류 정확도 : ', accuracy_score(testY, pred))
    print('중요변수 : ', model.feature_importances_)

"""
[Randomforest 문제2]
중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 

예제 파일 : https://github.com/pykwon  ==>  patient.csv

<변수설명>
  STA : 환자 생사 여부 (0:생존, 1:사망)
  AGE : 나이
  SEX : 성별
  RACE : 인종
  SER : 중환자 치료실에서 받은 치료
  CAN : 암 존재 여부
  INF : 중환자 치료실에서의 감염 여부
  CPR : 중환자 치료실 도착 전 CPR여부
  HRA : 중환자 치료실에서의 심박수
"""
def RFquestion2():
    data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/patient.csv")
    print(data.head(5))
    """
       ID  STA  AGE  SEX  RACE  SER  CAN  CRN  INF  CPR  HRA
        0   8    0   27    1     1    0    0    0    1    0   88
        1  12    0   39    0     1    0    0    0    0    0   80
        2  14    0   27    0     1    1    0    0    0    0   70
        3  28    0   54    0     1    0    0    0    1    0  103
        4  32    0   27    1     1    1    0    0    1    0  154
    """
    print(data.columns) #['ID', 'STA', 'AGE', 'SEX', 'RACE', 'SER', 'CAN', 'CRN', 'INF', 'CPR', 'HRA']
    print(data.shape)   #(200, 11)
    print(data.isnull().any())  #결측치 없음

    #종속변수 데이터프레임 만들기
    dataY = data['STA']

    #독립변수 데이터프레임 만들기
    dataX = data.drop(columns=['STA', 'ID'])

    #학습 데이터 분리
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.3, random_state=1)

    #모델 생성 : 랜덤포레스트
    model = RandomForestClassifier(criterion='entropy', n_estimators=500)
    model.fit(trainX, trainY)

    #모델 테스트해보기
    pred = model.predict(testX)
    print('예측값 : ', pred[:6])    #예측값 :  [1 0 0 0 1 1]
    print('실제값 : ', testY[:6])   #실제값 :  [0 0 0 0 1 1]
    print('맞춘 갯수 : ', sum(testY == pred))   #맞춘 갯수 :  54
    print('전체 대비 맞춘 비율 : ', sum(testY == pred) / len(testY))    #전체 대비 맞춘 비율 :  0.9
    print('분류 정확도 : ', accuracy_score(testY, pred))               #분류 정확도 :  0.9


RFquestion1()
RFquestion2()