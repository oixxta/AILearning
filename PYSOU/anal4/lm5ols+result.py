"""
ols가 제공하는 표에 대해 알아보기
"""
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(df.head(3))
print(df.corr())
#          친밀도     적절성     만족도
#친밀도  1.000000  0.499209  0.467145
#적절성  0.499209  1.000000  0.766853
#만족도  0.467145  0.766853  1.000000

#적절성과 만족도로 ols 표 만들기
#회귀분석 수행
import statsmodels.formula.api as smf
model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()
print(model.summary())
"""
<ols가 제공하는 표>
                            OLS Regression Results
==============================================================================
Dep. Variable:                  만족도   R-squared:                       0.588
Model:                            OLS   Adj. R-squared:                  0.586
Method:                 Least Squares   F-statistic:                     374.0
Date:                Fri, 22 Aug 2025   Prob (F-statistic):           2.24e-52
Time:                        16:10:33   Log-Likelihood:                -207.44
No. Observations:                 264   AIC:                             418.9
Df Residuals:                     262   BIC:                             426.0
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.7789      0.124      6.273      0.000       0.534       1.023
적절성          0.7393      0.038     19.340      0.000       0.664       0.815
==============================================================================
Omnibus:                       11.674   Durbin-Watson:                   2.185
Prob(Omnibus):                  0.003   Jarque-Bera (JB):               16.003
Skew:                          -0.328   Prob(JB):                     0.000335
Kurtosis:                       4.012   Cond. No.                         13.4
==============================================================================
절편(Intercept) : 0.7789, 독립변수 '적절성'의 기울기 : 0.7393.
위 표로 만든 수식 : y = 0.7393 * x + 0.7789

표준오차 : 동일한 모집단에서 반복하여 표본을 추출하는 경우, 얻을 수 있는 계수 추정치 간의 변동성을 추정한 값

R²(R-squared, 결정계수, 설명력) : 독립변수 x가 종속변수 y를 얼마나 설명하느냐를 판단, 값을 얼마나 설명하고 
있는지를 알려주기 때문에 설명력이라고 부름. 1에 가까울수록 좋으나 1에 너무 가까어지면 
기형보델(오버피팅)이 될 수 있음. 오버피팅될 경우, 기존 모델에선 쓰기 좋으나, 비슷한 다른 모델들에 적용하기엔
문제가 생길 수 있기 때문에 지양되어야 함. 상관계수를 제곱해서 결정계수를 구할 수도 있음.

Adj.R²(수정된 결정계수) : 독립변수가 두 개 이상일때 사용됨, 결정계수는 독립변수가 많아지면 
의미없는 독립변수가 들어가더라도 설명력이 좋아지기 때문에 기존 독립변수 대신 사용되어야 함.

Prob F-statistic(): F값으로 부터 나온 P값. 95% 신뢰 수준에서 0.05보다 작아야 의미있는 모델로 볼 수 있음.
OLS가 제공하는 표에서 가장 먼저 확인해야 하는 숫자임.

Durbin-Watson: 잔차들의 독립성을 확인하는 수치. 2에 최대한 가까울 수록 좋은 모델로 
볼 수 있음.(자기상관이 독립이다.)

자기상관 : 일반적으로 패턴이 반복되는 일련의 데이터에서 발생. 즉, 시계열 등의 자료에서 이들끼리 상관이나
회귀분석을 하는 경우에 두 시계열 내에 있는 자기상관 패턴끼리 상관관계가 나타날 수 있음.
시간 또는 공간적으로 연속된 일련의 관측치들 간에 존재하는 상관관계.

Jarque-Bera(JB) : 자큐베라, 적합도 검정으로 왜도와 첨도가 정규분포로 보기에 적합한지 확인하는 값. 
양수만 존재함. 0에 가까울수록 정규분포를 따름.

Prob(JB) : 프랍 자큐베라, 오차의 정규성 가정을 검증함. 0.05보다 작으면 정규성 실패.

Prob(Omnibus) : 회귀모형의 유의성 검증에 사용함. 0.05보다 작으면 유의하다고 볼 수 있음.

t-value = 기울기를 표준오차로 나눈 것. (B(기울기) / SE(표준오차)), 가설검정에서 쓰는 T값.

"""
print('parameters : ', model.params)
print('R squared : ', model.rsquared)
print('p value : ', model.pvalues)
#print('predicted value : ', model.predict())
print('실제값 : ', df.만족도[0], '\n예측값 : ', model.predict()[0])

import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='Malgun Gothic')
plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)
plt.plot(df.적절성, df.적절성 * slope + intercept, 'b')
plt.show()
plt.close()