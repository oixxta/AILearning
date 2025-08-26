#선형회귀 평가 지표 관련

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#공부 시간에 따른 시험 점수 데이터 생성하기 : 표본 16개
data = pd.DataFrame({'studytime' : [3, 4, 5, 8, 10, 5, 8, 6, 3, 6, 10, 9, 7, 0, 1, 2],
                     'score' : [76, 74, 74, 89, 66, 75, 84, 82, 73, 81, 95, 88, 83, 40, 70, 69]}) #16개의 데이터

print(data.head(3))

# dataset을 train데이터와 test데이터로 나눔 : train_test_split으로
# dataset은 절대로 sort를 하면 안됨!(시계열 데이터는 제외)
train, test = train_test_split(data, test_size=0.4, random_state=2) #train : test를 6:4로 데이터를 나눔. random_state : 시드고정
print(len(train), len(test))
x_train = train[['studytime']]
y_train = train['score']
x_test = train[['studytime']]
y_test = train['score']             #2차원으로 입력하고 1차원으로 출력하기 위해 리스트형으로 나눔.
print(x_train)
print(y_train)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(x_train, y_train) #모델 학습 시, 트레인데이터만 가지고 학습
y_pred = model.predict(x_test)  #모델 평가 시, 테스트데이터를 사용
print('예측값 : ', np.round(y_pred, 0)) #[76 68 73 71 80 73 83 87 78]
print('실제값 : ', y_test.values)       #[74 70 76 69 81 73 83 88 75]


#모델의 성능 평가 : R² & MSE, 모델의 성능평가는 옆의 두 가지가 일반적임.
#결정계수 수식으로 직접 작성 후 api 매소드로 나온 결과물과 비교:
#잔차 구하기
y_mean = np.mean(y_test)    # y의 평균값
#오차 제곱합 : sum(y관측값 - y예측값)²
bunja = np.sum(np.square(y_test - y_pred))
#편차 제곱합 : sum(y관측값 - y평균값)²
bunmo = np.sum(np.square(y_test - y_mean))
#1 - (오차제곱합 / 편차제곱합)
r2 = 1 - bunja / bunmo
print('계산에 의한 결정계수 : ', r2)   #0.9186441060519428


#api 제공 매소드와 비교
from sklearn.metrics import r2_score
print('api 제공 매소드 : ', r2_score(y_test, y_pred))   #0.9186441060519428
#r2_score는 의미없는 독립변수가 늘어나면 정확도가 떨어지게됨!


#R² 값은 분산을 기반으로 측정하는 도구인데 중심극한정리에 의해 표본데이터가 많아질수록 그 수치도 증가한다.
import seaborn as sns
import matplotlib.pyplot as plt

def linearFunc(df, test_size):
    train, test = train_test_split(data, train_size=test_size, shuffle=True, random_state=2)
    x_train = train[['studytime']]
    y_train = train['score']
    x_test = train[['studytime']]
    y_test = train['score']

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    #R² 계산
    print('R제곱 값 : ', r2_score(y_test, y_pred))
    print('test data 비율 : 전체 데이터 수의 {0}%'.format(test_size * 100))
    print('데이터 수 : {0}개'.format(x_train))

    #시각화
    sns.scatterplot(x=df['studytime'], y=df['score'], color='green')
    sns.scatterplot(x=x_test['studytime'], y=y_test, color='red')
    sns.lineplot(x=x_test['studytime'], y=y_pred, color='blue')
    plt.show()
    plt.close()

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  #test 자료수를 10%에서 50%로 늘려가며 R²값 구하기.
for i in test_sizes:
    linearFunc(data, i)