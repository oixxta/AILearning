"""
ML(기계학습 - 지도학습)
회귀분석 : 입력 데이터에 대한 잔차 제곱합이 최소가 되는 추세선(회귀선)을 만들고,
이를 통해 독립변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석하는 것.
독립변수 : 연속형, 종속변수 : 연속형. 두 변수는 상관관계가 있어야 하며, 인과관계를 보여야 한다.
정량적인 모델을 생성
"""
import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
np.random.seed(12)

# 모델 생성 후 맛보기
# 정량적 모델
# 방법 1 : make_regression을 사용 : 모델을 만들진 않음, 확인용
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True) #샘플 50개,    기울기 출력 여부 
print(x)
print(y)
print(coef) #89.47430739278907 : 기울기
#위 모델을 바탕으로 만든 회귀식 : y = 89.47430739278907 * x + 100
#해당 회귀식으로 실제 x값 넣어서 비교해보기
y_pred = 89.47430739278907 * -1.70073563 + 100
print(y_pred)   #-52.17214255248879, 실제 값과 거의 비슷하게 나옴.
#미지의 x값 5에 대한 예측값 y 구하기.
print('y_pred_new : ', 89.47430739278907 * 5 + 100) #547.3715369639453


# 방법 2 : Linear regression을 사용 : model 존재
from sklearn.linear_model import LinearRegression
model = LinearRegression()
xx = x
yy = y
fit_model = model.fit(xx, yy)   #학습 데이터로 모형 추정하기. 절편과 기울기 얻기
print(fit_model.coef_)  #기울기 출력    : 89.47430739
print(fit_model.intercept_) #절편 출력  : 100.0
#위 결과로 회귀식 만들기 : y = 89.47430739 * xx + 100.0
#xx의 0번째 자료를 넣어서 확인해보기 
print(89.47430739 * xx[[0]] + 100.0)    #-52.17214291, 실제 값(-52.17214291)과 거의 비슷한 값으로 나옴.
print('예측값 y[0] : ', model.predict(xx[[0]])) #-52.17214291
#xx에 미지의 값 5에 대한 예측값 얻기
print(model.predict([[5]]))   #547.37153696
print(model.predict([[5], [3]]))    #547.37153696 368.42292218


# 방법 3 : ols를 사용
import statsmodels.formula.api as smf
import pandas as pd
x1 = xx.flatten()         #x1을 독립변수로 사용 차원 축소, xx.revel()도 사용 가능.
print(x1.shape)
y1 = yy                  #y1을 종속변수로 사용

data = np.array([x1, y1])
print(data.T)
df = pd.DataFrame(data = data.T, columns=['x1', 'y1'])
print(df.head(2))
model2 = smf.ols(formula='y1 ~ x1', data=df).fit()
print(model2.summary())
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                     y1   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.905e+32
Date:                Fri, 22 Aug 2025   Prob (F-statistic):               0.00
Time:                        11:58:57   Log-Likelihood:                 1460.6
No. Observations:                  50   AIC:                            -2917.
Df Residuals:                      48   BIC:                            -2913.
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    100.0000   7.33e-15   1.36e+16      0.000     100.000     100.000
x1            89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474
==============================================================================
Omnibus:                        7.616   Durbin-Watson:                   1.798
Prob(Omnibus):                  0.022   Jarque-Bera (JB):                8.746
Skew:                           0.516   Prob(JB):                       0.0126
Kurtosis:                       4.770   Cond. No.                         1.26
==============================================================================

절편(Intercept) : 100, 독립변수 x1의 기울기 : 89.4743
"""
#위 결과로 만든 수식 : y = 89.4743 * x + 100.
print(x1[:2])   #-1.70073563 -0.67794537
#기존 데이터를 예측해보기
new_df = pd.DataFrame({'x1' : [-1.70073563, -0.67794537]})
new_pred = model2.predict(new_df)
print(new_pred.values) #0   -52.172143, 1    39.341308. 정상적으로 실제값과 비슷한 값을 반환함.
#새로운 데이터(123, 5), 독립변수로 종속변수 예측해보기.
new_df2 = pd.DataFrame({'x1' : [123, 5]})
new_pred2 = model2.predict(new_df2)
print(new_pred2.values) # 11105.33980931   547.37153696

