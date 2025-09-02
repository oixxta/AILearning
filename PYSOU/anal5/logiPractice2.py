"""
[로지스틱 분류분석 문제3]

Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
얘를 사용해도 됨   'testdata/advertisement.csv' 
참여 칼럼 : 
   - Daily Time Spent on Site : 사이트 이용 시간 (분)
   - Age : 나이,
   - Area Income : 지역 소득,
   - Daily Internet Usage :일별 인터넷 사용량(분),
   - Clicked on Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
새로운 데이터로 분류 작업을 진행해 본다.


독립변수 : Daily Time Spent on Site, Age, Area Income, Daily Internet Usage
종속변수 : Clicked on Ad
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#데이터 긁어오기 및 사용하기 좋게 가공하기
originData = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/advertisement.csv")
print(originData.head(3))
print(originData.info())
"""
 0   Daily Time Spent on Site  1000 non-null   float64
 1   Age                       1000 non-null   int64
 2   Area Income               1000 non-null   float64
 3   Daily Internet Usage      1000 non-null   float64
 4   Ad Topic Line             1000 non-null   object
 5   City                      1000 non-null   object
 6   Male                      1000 non-null   int64
 7   Country                   1000 non-null   object
 8   Timestamp                 1000 non-null   object
 9   Clicked on Ad             1000 non-null   int64
"""
data = originData.drop(['Ad Topic Line', 'City', 'Male', 'Country', 'Timestamp'], axis=1)
print(data.head(3))
"""
   Daily Time Spent on Site  Age  Area Income  Daily Internet Usage  Clicked on Ad
0                     68.95   35     61833.90                256.09              0
1                     80.23   31     68441.85                193.77              0
2                     69.47   26     59785.94                236.50              0
"""
print(data['Clicked on Ad'].unique())   #Clicked on Ad 전체에 0과 1만 있는지 재확인, 이상무.

xData = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
yData = data['Clicked on Ad']

#학습데이터 / 연습데이터 분리 (8:2 비율로 하겠음.)
trainDataX, testDataX, trainDataY, testDataY = train_test_split(xData, yData, test_size=0.2, random_state=1)
print(trainDataX.shape, testDataX.shape, trainDataY.shape, testDataY.shape)


#표준화 작업 : 변수들의 단위와 범위가 달라 모델 학습에 불리할 수 있으니 표준화시킴.
scaler = StandardScaler()
trainDataX = scaler.fit_transform(trainDataX)
testDataX = scaler.transform(testDataX)


#로지스틱 회귀 모델 학습
model = LogisticRegression().fit(trainDataX, trainDataY)


#모델 테스트 : 분류 예측
#수작업, 실제값과 예측값 10개씩만 비교해보기
yPred = model.predict(testDataX)
print("예측값 : ", yPred[:10])          #[1 0 0 0 0 1 1 1 0 1]
print("실제값 : ", testDataY[:10])      #[1 0 0 0 0 1 1 1 0 1]
#매서드를 쓴 자동화
f_value = model.decision_function(testDataX)


#모델 성능 출력
print(confusion_matrix(testDataY, yPred))   #컨퓨션 매트릭스 보기
"""
[[102   1]
 [  8  89]]     맞춘 갯수 : 191개 , 틀린 갯수 : 9개, 전체 데이터 200개
"""
from sklearn import metrics
print('acc(정확도) : ', metrics.accuracy_score(testDataY, yPred))
print('recall(재현률) : ', metrics.recall_score(testDataY, yPred))
print('precision(정밀도) : ', metrics.precision_score(testDataY, yPred))


#ROC 커브 시각화 하기
fpr, tpr, thresholds = metrics.roc_curve(testDataY, model.decision_function(testDataX))
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier line')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()
plt.close()


#AUC 계산
print('AUC : ', metrics.auc(fpr, tpr))  #0.9727754979481532
#AUC 값은 1에 가까울수록 좋은 모델임. 위의 경우 양호함.


#위에서 만든 모델로 새로운 데이터를 분류시켜보기
newData = pd.DataFrame([[66, 5, 0, 220]], columns=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'])
newData_scaled = scaler.transform(newData)
newPred = model.predict(newData_scaled)
print("새로운 사용자의 광고 클릭 예측 (1: 클릭함, 0: 클릭안함):", newPred[0])

"""
새로운 사용자의 광고 클릭 예측 (1: 클릭함, 0: 클릭안함): 1

소득이 없는 5살 인터넷 중독 꼬마의 광고 클릭 확률 : 클릭함.
"""