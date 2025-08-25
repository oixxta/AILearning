"""
단순 선형회귀 : OLS 사용.
상관관계가 선형회귀모델에 미치는 영향에 대해.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
print(iris.head(2))
print(iris.iloc[:, 0:4].corr())
#상관관계 출력 결과 : 
#              sepal_length  sepal_width  petal_length  petal_width
#sepal_length      1.000000    -0.117570      0.871754     0.817941
#sepal_width      -0.117570     1.000000     -0.428440    -0.366126
#petal_length      0.871754    -0.428440      1.000000     0.962865
#petal_width       0.817941    -0.366126      0.962865     1.000000


# 연습 1) 상관관계가 약한(-0.117570) 두 변수(sepal_width, sepal_length)를 사용
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print('검정 결과1 : ', result1.summary())
"""
검정 결과1 :                              OLS Regression Results
==============================================================================
Dep. Variable:           sepal_length   R-squared:                       0.014
Model:                            OLS   Adj. R-squared:                  0.007
Method:                 Least Squares   F-statistic:                     2.074
Date:                Mon, 25 Aug 2025   Prob (F-statistic):              0.152
Time:                        11:04:13   Log-Likelihood:                -183.00
No. Observations:                 150   AIC:                             370.0
Df Residuals:                     148   BIC:                             376.0
Df Model:                           1
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept       6.5262      0.479     13.628      0.000       5.580       7.473
sepal_width    -0.2234      0.155     -1.440      0.152      -0.530       0.083
==============================================================================
Omnibus:                        4.389   Durbin-Watson:                   0.952
Prob(Omnibus):                  0.111   Jarque-Bera (JB):                4.237
Skew:                           0.360   Prob(JB):                        0.120
Kurtosis:                       2.600   Cond. No.                         24.2
==============================================================================

상관계수의 값이 너무 안좋음, 좋지 않은 모델.
"""
print('결정계수 R² : ', result1.rsquared)   # 0.13822
print('p값 : ', result1.pvalues.iloc[1]) # 0.15189826071144785, 0.05보다 큼, 유의하지 않은 모델.
#plt.scatter(iris.sepal_width, iris.sepal_length)
#plt.plot(iris.sepal_width, result1.predict(), color='r')
#plt.show()
#plt.close()


# 연습 2) 상관관계가 강한(0.871754) 두 변수(sepal_length, petal_length)를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print('검정 결과2 : ', result2.summary())
"""
검정 결과2 :                              OLS Regression Results
==============================================================================
Dep. Variable:           petal_length   R-squared:                       0.184
Model:                            OLS   Adj. R-squared:                  0.178
Method:                 Least Squares   F-statistic:                     33.28
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           4.51e-08
Time:                        11:11:57   Log-Likelihood:                -282.38
No. Observations:                 150   AIC:                             568.8
Df Residuals:                     148   BIC:                             574.8
Df Model:                           1
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept       9.0632      0.929      9.757      0.000       7.227      10.899
sepal_width    -1.7352      0.301     -5.768      0.000      -2.330      -1.141
==============================================================================
Omnibus:                        7.319   Durbin-Watson:                   0.602
Prob(Omnibus):                  0.026   Jarque-Bera (JB):                3.729
Skew:                           0.136   Prob(JB):                        0.155
Kurtosis:                       2.277   Cond. No.                         24.2
==============================================================================
"""
print('결정계수 R² : ', result2.rsquared)   #0.7599546457725151
print('p값 : ', result2.pvalues.iloc[1])# 1.0386674194499307e-47, 0.05보다 작음, 유의한 모델
#plt.scatter(iris.petal_length, iris.sepal_length)
#plt.plot(iris.petal_length, result2.predict(), color='r')
#plt.show()
#plt.close()


# 일부의 실제값과 예측값 비교
print('실제값 : ', iris.sepal_length[:5].values)    #[5.1 4.9 4.7 4.6 5. ]
print('예측값 : ', result2.predict()[:5])           #[4.8790946  4.8790946  4.83820238 4.91998683 4.8790946 ]
print()


# 새로운 값으로 예측
newData = pd.DataFrame({'petal_length':[1.1, 0.5, 5.0]})
y_pred = result2.predict(newData)
print('예측 결과(sepal_length) : ', y_pred)
# 예측 결과(sepal_length) : 
# 0    4.756418
# 1    4.511065
# 2    6.351215

#꽃잎의 길이로 꽃받힘의 길이 예측 성공


# 다중 선형회귀 : 독립변수가 복수인.
#result3 = smf.ols(formula='sepal_length ~ petal_length+petal_width+sepal_width', data=iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_length', 'species']))
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
print(result3.summary())
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:           sepal_length   R-squared:                       0.859
Model:                            OLS   Adj. R-squared:                  0.856
Method:                 Least Squares   F-statistic:                     295.5
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.59e-62
Time:                        11:42:46   Log-Likelihood:                -37.321
No. Observations:                 150   AIC:                             82.64
Df Residuals:                     146   BIC:                             94.69
Df Model:                           3
Covariance Type:            nonrobust
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        1.8560      0.251      7.401      0.000       1.360       2.352
petal_length     0.7091      0.057     12.502      0.000       0.597       0.821
petal_width     -0.5565      0.128     -4.363      0.000      -0.809      -0.304
sepal_width      0.6508      0.067      9.765      0.000       0.519       0.783
==============================================================================
Omnibus:                        0.345   Durbin-Watson:                   2.060
Prob(Omnibus):                  0.842   Jarque-Bera (JB):                0.504
Skew:                           0.007   Prob(JB):                        0.777
Kurtosis:                       2.716   Cond. No.                         54.7
==============================================================================

독립변수가 다수이기에 Adj. R-squared로 확인해야함, 좋은 모델임.
Prob (F-statistic)값이 0.05보다 작고, 0에 가까움, 아주 좋은 모델임.
세 개의 모델 모두 p의 값이 0에 가까움, 유이한 모델임.

"""