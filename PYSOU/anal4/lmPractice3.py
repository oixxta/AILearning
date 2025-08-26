import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import sys
import pickle
import MySQLdb


"""
회귀분석 문제 4) 

원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
장고로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.
"""
def lmPractice4():
    #데이터 긁어오기 : 서버에서
    try:
        with open('myMaria.dat', mode='rb') as obj:
            config = pickle.load(obj)
    except Exception as e:
        print('fail to read server!', e)
        sys.exit()
    
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()

        sql = """
            SELECT * FROM jikwon
        """
        cursor.execute(sql)
        dataFromDb = pd.DataFrame(cursor.fetchall(), columns=['jikwonno' , 'jikwonname' , 'busernum' , 'jikwonjik' , 'jikwonpay' , 'jikwonibsail' , 'jikwongen' , 'jikwonrating'])
    except Exception as e:
        print('fail to read server!', e)
    finally:
        cursor.close()
        conn.close()
    
    print(dataFromDb.head(5))
 #필요없는 칼럼 제거
    #입사일 칼럼을 바탕으로 연차 칼럼 만들기
    


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


lmPractice4()
#lmPractice5()
