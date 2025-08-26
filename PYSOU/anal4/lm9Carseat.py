"""
회귀분석 문제 3)    
kaggle.com에서 carseats.csv 파일을 다운 받아 Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
회귀분석모형의 적절성을 위한 조건도 체크하시오.
완성된 모델로 Sales를 예측.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')

# 데이터 가져오기
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv")
print(data)
print(data.info)


# 데이터 상관관계 확인하기
data.drop([data.columns[6],data.columns[9],data.columns[10]], axis=1, inplace=True)
print(data.corr())
"""
               Sales  CompPrice    Income  Advertising  Population     Price       Age  Education
Sales        1.000000   0.064079  0.151951     0.269507    0.050471 -0.444951 -0.231815  -0.051955
CompPrice    0.064079   1.000000 -0.080653    -0.024199   -0.094707  0.584848 -0.100239   0.025197
Income       0.151951  -0.080653  1.000000     0.058995   -0.007877 -0.056698 -0.004670  -0.056855
Advertising  0.269507  -0.024199  0.058995     1.000000    0.265652  0.044537 -0.004557  -0.033594
Population   0.050471  -0.094707 -0.007877     0.265652    1.000000 -0.012144 -0.042663  -0.106378
Price       -0.444951   0.584848 -0.056698     0.044537   -0.012144  1.000000 -0.102177   0.011747
Age         -0.231815  -0.100239 -0.004670    -0.004557   -0.042663 -0.102177  1.000000   0.006488
Education   -0.051955   0.025197 -0.056855    -0.033594   -0.106378  0.011747  0.006488   1.000000
"""

# 단순선형회귀 모델 만들기 : 종속변수 : Sales, 독립변수 : Income, Advertising, Price, Age 사용.
lModel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=data).fit()
print(lModel.summary())
"""
                           OLS Regression Results
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.371
Model:                            OLS   Adj. R-squared:                  0.364
Method:                 Least Squares   F-statistic:                     58.21
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.33e-38
Time:                        16:49:27   Log-Likelihood:                -889.67
No. Observations:                 400   AIC:                             1789.
Df Residuals:                     395   BIC:                             1809.
Df Model:                           4
Covariance Type:            nonrobust
===============================================================================
                  coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      15.1829      0.777     19.542      0.000      13.656      16.710
Income          0.0108      0.004      2.664      0.008       0.003       0.019
Advertising     0.1203      0.017      7.078      0.000       0.087       0.154
Price          -0.0573      0.005    -11.932      0.000      -0.067      -0.048
Age            -0.0486      0.007     -6.956      0.000      -0.062      -0.035
==============================================================================
Omnibus:                        3.285   Durbin-Watson:                   1.931
Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.336
Skew:                           0.218   Prob(JB):                        0.189
Kurtosis:                       2.903   Cond. No.                     1.01e+03
==============================================================================

Adj. R-squared 이상 무 (0.3 ~ 0.5 어느 정도 경향 설명 가능 및 오버피팅 발생하지 않음.)
F-statistic 이상 무 (0.05보다 작음.)
모든 독립변수 P값 이상 무 (전부 0.05보다 작음.)
Durbin-Watson 이상 무 (2에 근사함.)
"""

# 만들어진 모델의 저장 후, 읽어서 사용하기.
# 방법 1 : pickle 모듈 사용
"""
import pickle
with open('myModel.pickle', mode='wb') as obj:     #저장
    pickle.dump(lModel, obj)
with open('myModel.pickle', mode='rb') as obj:     #읽기
    myModel = pickle.load(obj)
myModel.predict('~~~~')                            #사용
"""
# 방법 2 : LIB 모듈 사용
"""
import joblib
joblib.dump(lModel, 'myModel.model')               #저장
myModel = joblib.load('myModel.model')             #읽기
myModel.predict('~~~~')                            #사용
"""

# 선형회귀분석의 기존 가정 충족 조건 순응 확인
data_lm = data.iloc[:,[0,2,3,5,6]]
#잔차(예측값과 실제값의 차이) 구하기
fitted = lModel.predict(data_lm)
residual = data_lm['Sales'] - fitted
print('실제값 : ', data_lm['Sales'][:5].values)   #[ 9.5  11.22 10.06  7.4   4.15]
print('예측값 : ', fitted[:5].values)             #[8.37832076 9.71025253 9.31205018 8.51137246 7.05543592]
print('잔차값 : ', residual[:5].values)           #[ 1.12167924  1.50974747  0.74794982 -1.11137246 -2.90543592]
print('잔차의 평균값 : ', np.mean(residual))       #-1.4921397450962103e-15, 0에 가까움.

#1. 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화하는지 확인하기
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='gray')
plt.show()  #확인 결과, 잔차의 추세선(붉은선)이 파선(---)에 가까움, 선형성 충족함.
plt.close()

#2. 정규성 : 잔차항(오차항)이 정규분포를 따르는지 확인하기
import scipy.stats as stats
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x=x, y=y)
plt.plot([-3, 3], [-3, 3], '--', color='gray')
plt.show()  #확인 결과, 정규성 만족
plt.close()
from scipy.stats import shapiro
stat, pv = shapiro(residual)
print(f"shapiro-wilk test로 정규성 확인 : 통계량 : {stat:.4f}, p-value:{pv:.4f}")   #통계량 : 0.9949, p-value:0.2127
print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성이 있음')  #정규성 만족

#3. 독립성 : 독립변수의 값이 서로 관련되지 않는지 확인하기.
print(lModel.summary())
#확인 결과, Durbin-Watson 값이 1.931이 나옴. 자기상관이 없음, 독립성 충족함.
import statsmodels.api as sm
print('Durbin-Watson : ', sm.stats.stattools.durbin_watson(residual))   #1.931498127082959


#4. 등분산성(Homoscedasticity) : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.(독립변수가 복수일 때만 한다)
#시각화 : 잔차가 일정한 분포를 갖고 분산을 갖고 퍼져있어야 함.
sr = stats.zscore(residual)
sns.regplot(x=fitted, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color':'red'})
plt.show()  #시각화로 확인
#수치로 확인 : het_breuschpagan
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, lModel.model.exog) #잔차와 독립변수의 관계를 확인함
print(bp_test[0], bp_test[1])  #통계량 출력 : 1.1276773825795061, p값 : 0.8898563862439574
#p값이 0.05보다 커야 등분산성이 만족함.
#등분산성 만족(0.8898563862439574 > 0.05)


#5. 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.(독립변수가 복수일 때만 한다)
#VIF(분산팽창지수)로 다중공선성 여부를 확인 : 연속형의 경우, 10을 넘으면 다중공선성을 의심해야 함.
from statsmodels.stats.outliers_influence import variance_inflation_factor
imsiDf = data[['Income', 'Advertising', 'Price', 'Age']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(imsiDf, i) for i in range(imsiDf.shape[1])]
print(vifdf)
#   vif_value
#0   5.971040       Income
#1   1.993726       Advertising
#2   9.979281       Price
#3   8.267760       Age
#4개의 독립변수가 모두 VIF값 10을 넘지 않았기 때문에, 다중공선성 문제가 발생하지 않는다.


# 저장된 모델을 읽어 새로운 데이터에 대한 예측
import joblib
ourModel = joblib.load('myModel.model')
newDf = pd.DataFrame({'Income':[35, 63, 25], 'Advertising':[6, 3, 11], 'Price':[105, 88, 77], 'Age':[33, 55, 22]})
new_pred = ourModel.predict(newDf)
print('Sales 예측 결과 :\n', new_pred)
# Sales 예측 결과 :
# 0     8.664248
# 1     8.507928
# 2    11.296480