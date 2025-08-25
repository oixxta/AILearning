"""
회귀분석 문제 2) 
testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.

  - 국어 점수를 입력하면 수학 점수 예측
  - 국어, 영어 점수를 입력하면 수학 점수 예측
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

#데이터 가져오기
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv")
print(data.head(3))

#상관관계 출력
print(data.iloc[:, 1:4].corr())
#          국어        영어        수학
#국어  1.000000   0.915188   0.766263
#영어  0.915188   1.000000   0.809668
#수학  0.766263   0.809668   1.000000

#수학점수를 종속변수로, 국어점수를 독립변수로 한 모델(두 모델은 상관관계가 강함.)
result1 = smf.ols(formula='수학 ~ 국어', data=data).fit()
print(result1.summary())
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                    수학   R-squared:                       0.587
Model:                            OLS   Adj. R-squared:                  0.564
Method:                 Least Squares   F-statistic:                     25.60
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.16e-05
Time:                        12:31:07   Log-Likelihood:                -76.543
No. Observations:                  20   AIC:                             157.1
Df Residuals:                      18   BIC:                             159.1
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     32.1069      8.628      3.721      0.002      13.981      50.233
국어            0.5705      0.113      5.060      0.000       0.334       0.807
==============================================================================
Omnibus:                        1.833   Durbin-Watson:                   2.366
Prob(Omnibus):                  0.400   Jarque-Bera (JB):                0.718
Skew:                          -0.438   Prob(JB):                        0.698
Kurtosis:                       3.310   Cond. No.                         252.
==============================================================================

"""
#모델 시각화 확인
plt.scatter(data.국어, data.수학)
plt.xlabel('korean')
plt.ylabel('math')
plt.plot(data.국어, result1.predict(), color='r')
plt.show()
plt.close()

#점수예측 : 국어점수가 60점일때 수학 점수
print(0.5705 * 60 + 32.1069)    #66.33690000000001
print(result1.predict(pd.DataFrame({'국어' : [60]})))   # 66.339908



#수학점수를 종속변수로, 국어점수와 영어점수를 독립변수로 한 모델(세 모델은 상관관계가 강함.)
result2 = smf.ols(formula='수학 ~ 국어 + 영어', data=data).fit()
print(result2.summary())
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                    수학   R-squared:                       0.659
Model:                            OLS   Adj. R-squared:                  0.619
Method:                 Least Squares   F-statistic:                     16.46
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           0.000105
Time:                        12:34:48   Log-Likelihood:                -74.617
No. Observations:                  20   AIC:                             155.2
Df Residuals:                      17   BIC:                             158.2
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     22.6238      9.482      2.386      0.029       2.618      42.629
국어           0.1158      0.261      0.443      0.663      -0.436       0.667
영어           0.5942      0.313      1.900      0.074      -0.066       1.254
==============================================================================
Omnibus:                        6.313   Durbin-Watson:                   2.163
Prob(Omnibus):                  0.043   Jarque-Bera (JB):                3.824
Skew:                          -0.927   Prob(JB):                        0.148
Kurtosis:                       4.073   Cond. No.                         412.
==============================================================================
"""

#점수예측 : 국어, 영어점수가 각각 60점일때 수학 점수
print(result2.predict(pd.DataFrame({'국어' : [60], '영어' : [60]})))    #65.224234