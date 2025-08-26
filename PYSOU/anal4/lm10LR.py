# sklearn 모듈의 LinearRegression 클래스 사용
# sklearn모듈은 주로 입력은 2차원, 출력은 1차원으로 만들어짐.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pylab as plt

#StandardScaler : 표준화
#MinMaxScaler   : 정규화


sampleSize = 100
np.random.seed(1)

# 편차가 없는 데이터 생성
x = np.random.normal(0, 10, sampleSize)                 #독립변수
y = np.random.normal(0, 10, sampleSize) + x * 30        #종속변수
print(x[:5])    
print(y[:5])    

scaler = MinMaxScaler()     #정규화, 스케일링 : 모델의 성능을 향상시키기 위한 방법 중 하나.
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled[:5])
#plt.scatter(x_scaled, y)
#plt.show()
#plt.close()

# 상관계수 확인
print(np.corrcoef(x, y))    #0.99939357, 상관계수 매우 좋음.
#plt.scatter(x, y)
#plt.show()
#plt.close()

# 모델 만들기
model  = LinearRegression().fit(x_scaled, y)
print(model)
print('계수 : ', model.coef_)       # 회귀계수 (각 독립변수가 종속변수에 미치는 영향), [1350.4161554]
print('절편 : ', model.intercept_)  # 절편, -691.1877661754081
print('결정계수 : ', model.score(x_scaled, y))  # 설명력, 훈련데이터 기준, 0.9987875127274646
# 수식 y = wx + b, y = 1350.4161554 * x + (-691.1877661754081)

# 모델 확인하기
y_pred = model.predict(x_scaled)
print('예측값 ŷ', y_pred[:5])   #[ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
print('실제값 y', y[:5])        #[ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]
#sklearn은 OLS와 달리 summary가 없음!

#선형회귀의 평가지표는 MAE, MSE, RMSE, R² 4가지가 있음
#선형회귀 모델 성능 파악용 함수 작성
def RegScoreFnc(y_real, y_pred):
    print('R²_score(결정계수) : {}'.format(r2_score(y_real, y_pred)))           #0.9987875127274646
    print('설명분산점수 : {}'.format(explained_variance_score(y_real, y_pred)))  #0.9987875127274646
    print('MSE (평균제곱오차) : {}'.format(mean_squared_error(y_real, y_pred)))  #86.14795101998747


RegScoreFnc(y, y_pred)


# 편차가 꽤 많이 있는 데이터 생성
x = np.random.normal(0, 1, sampleSize)                   #독립변수
y = np.random.normal(0, 500, sampleSize) + x * 30        #종속변수
print(x[:5])    
print(y[:5])
print('상관계수 : ', x_scaled, y)

scaler = MinMaxScaler()     #정규화, 스케일링 : 모델의 성능을 향상시키기 위한 방법 중 하나.
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled[:5])
plt.scatter(x_scaled, y)
plt.show()
plt.close()

# 상관계수 확인
print(np.corrcoef(x, y))    #0.00401167, 상관계수 매우 안좋음.

# 모델 만들기
model2  = LinearRegression().fit(x_scaled, y)
print(model2)
y_pred = model2.predict(x_scaled)

# 만든 모델 평가
RegScoreFnc(y, y_pred)