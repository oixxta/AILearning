import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

"""
회귀분석 문제 5) 

Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
testdata 폴더 : Consumo_cerveja.csv
Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
feature : Temperatura Media (C) : 평균 기온(C)
          Precipitacao (mm) : 강수(mm)

label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오

조건 : NaN이 있는 경우 삭제!
"""
def lmPractice5():
    #데이터 긁어오기
    data = pd.read_csv("Consumo_cerveja.csv", sep=',', skip_blank_lines=True, decimal=",")
    data.dropna(inplace=True)   #NAN 제거
    data.info()
    """
    #   Column                       Non-Null Count  Dtype
    ---  ------                       --------------  -----
    0   Data                         365 non-null    object
    1   Temperatura Media (C)        365 non-null    float64   #평균기온
    2   Temperatura Minima (C)       365 non-null    float64   #최저기온
    3   Temperatura Maxima (C)       365 non-null    float64   #최고기온
    4   Precipitacao (mm)            365 non-null    float64   #강수량
    5   Final de Semana              365 non-null    float64   #주말
    6   Consumo de cerveja (litros)  365 non-null    object    #맥주소비량
    """
    data['Consumo de cerveja (litros)'] = data['Consumo de cerveja (litros)'].astype(float) #맥주소비량을 float 자료형으로 변경
    data.drop([data.columns[0],data.columns[2],data.columns[3],data.columns[5]], axis=1, inplace=True)    #안 쓸 칼럼들 드랍
    print(data.head(3))


    #데이터 상관계수 확인 : 종속변수(y) Consumo de cerveja (litros), 독립변수(x) Temperatura Media (C),  Precipitacao (mm)
    print(data.corr())
    """
                                Temperatura Media (C)  Precipitacao (mm)  Consumo de cerveja (litros)
    Temperatura Media (C)                     1.000000           0.024416                     0.574615
    Precipitacao (mm)                         0.024416           1.000000                    -0.193784
    Consumo de cerveja (litros)               0.574615          -0.193784                     1.000000
    """

    #dataset을 train데이터와 test데이터로 나눔 : train_test_split으로
    train, test = train_test_split(data, test_size=0.7, random_state=1)
    print(len(train), len(test))
    x_train = train[['Temperatura Media (C)', 'Precipitacao (mm)']]
    y_train = train['Consumo de cerveja (litros)']
    x_test = test[['Temperatura Media (C)', 'Precipitacao (mm)']]
    y_test = test['Consumo de cerveja (litros)']


    #모델 학습 및 테스트
    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('예측값 : ', np.round(y_pred, 0)) #[26.    19.    25.    21.    30.    25.    29.   ]
    print('실제값 : ', y_test.values)       #[24.834 21.294 23.898 19.463 29.569 21.662 25.343]


    #만든 모델의 성능 평가 : R²(결정계수)
    print(r2_score(y_test, y_pred))     #0.3600695581190675

    #만든 모델의 성능 평가 : MSE
    print(mean_squared_error(y_test, y_pred))   #12.061743296255045


"""
다항회귀분석 문제) 

데이터 로드 (Servo, UCI) : "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data"
cols = ["motor", "screw", "pgain", "vgain", "class"]

 - 타깃/피처 (숫자만 사용: pgain, vgain)
   x = df[["pgain", "vgain"]].astype(float)   
   y = df["class"].values
 - 학습/테스트 분할 ( 8:2 )
 - 스케일링 (StandardScaler)
 - 다항 특성 (degree=2) + LinearRegression 또는 Ridge 학습
 - 성능 평가 
 - 시각화
"""
def lmPractice6():
    # 데이터 불러오기
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data")
    data.columns = ["motor", "screw", "pgain", "vgain", "class"]
    print(data.info())
    print(data.head(3))

    # 숫자형 피처와 타깃 추출
    x = data[["pgain", "vgain"]].astype(float)
    y = data["class"].astype(float)

    # 학습/테스트 분할 (8:2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # 스케일링
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 다항 특성 변환 (degree=2)
    poly = PolynomialFeatures(degree=2)
    x_train_poly = poly.fit_transform(x_train_scaled)
    x_test_poly = poly.transform(x_test_scaled)

    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # 예측
    y_pred = model.predict(x_test_poly)

    # 성능 평가
    print('R²: ', r2_score(y_test, y_pred))             #0.25524898311893973
    print('MSE: ', mean_squared_error(y_test, y_pred))  #1.2277575517112385

    # 시각화: 예측값 vs 실제값
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("real")
    plt.ylabel("pred")
    plt.title("real vs pred")
    plt.grid(True)
    plt.show()



#lmPractice5()
lmPractice6()