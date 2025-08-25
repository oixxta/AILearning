"""
*** 선형회귀분석의 기존 가정 충족 조건 ***
. 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
. 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
. 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.
. 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
. 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.

선형회귀분석 - OLS를 사용.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')

#Advertising-dataset 사용 : 각 매체의 광고비에 따른 판매량 관련 자료.
# 데이터 긁어오기 및 확인
advdf = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv', usecols=[1,2,3,4])
print(advdf.head(3))
print(advdf.shape)  #(200, 4)
print(advdf.index)
print(advdf.columns)
print(advdf.info())

# 상관관계 확인하기
print(advdf.corr())
#                 tv     radio  newspaper     sales
#tv         1.000000  0.054809   0.056648  0.782224
#radio      0.054809  1.000000   0.354104  0.576223
#newspaper  0.056648  0.354104   1.000000  0.228299
#sales      0.782224  0.576223   0.228299  1.000000
#경고!) 라디오와 신분의 상관관계가 너무 약함.

# 단순선형회귀 사용 : x, 독립변수:tv, y, 종속변수:sales
lModel1 = smf.ols(formula='sales ~ tv', data=advdf).fit()
print(lModel1.summary())
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.612
Model:                            OLS   Adj. R-squared:                  0.610
Method:                 Least Squares   F-statistic:                     312.1
Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.47e-42
Time:                        15:02:58   Log-Likelihood:                -519.05
No. Observations:                 200   AIC:                             1042.
Df Residuals:                     198   BIC:                             1049.
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
tv             0.0475      0.003     17.668      0.000       0.042       0.053
==============================================================================
Omnibus:                        0.531   Durbin-Watson:                   1.935
Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
Skew:                          -0.089   Prob(JB):                        0.716
Kurtosis:                       2.779   Cond. No.                         338.
==============================================================================
모든 값이 다 잘 나옴.
"""
print(lModel1.params)   #0.047537
print(lModel1.pvalues)  #1.406300e-35
print(lModel1.rsquared) #0.611875050850071


# 기존 데이터를 사용한 예측값
xPart = pd.DataFrame({'tv' : advdf.tv[:3]})
print('실제값 : ', advdf.sales[:3])                  #[22.1  10.4  9.3]
print('예측값 : ', lModel1.predict(xPart).values)    #[17.97077451  9.14797405  7.85022376]


# 새로운 데이터를 사용한 예측값
xNew = pd.DataFrame({'tv' : [100, 300, 500]})
print('예측값 : ', lModel1.predict(xNew).values)     #[11.78625759 21.29358568 30.80091377]


# 시각화 해보기
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('TV')
plt.ylabel('Sales')
y_pred = lModel1.predict(advdf.tv)
plt.plot(advdf.tv, y_pred, c = 'red')
plt.grid()
plt.show()
plt.close()


# 선형회귀분석의 기존 가정 충족 조건 충족 여부 확인
#잔차(예측값과 실제값의 차이) 구하기
fitted = lModel1.predict(advdf)
#print(fitted)
residual = advdf['sales'] - fitted
print('실제값 : ', advdf['sales'][:5].values)   #[22.1 10.4  9.3 18.5 12.9]
print('예측값 : ', fitted[:5].values)           #[17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
print('잔차값 : ', residual[:5].values)         #[4.12922549  1.25202595  1.44977624  4.26560543 -2.72721814]
print('잔차의 평균값 : ', np.mean(residual))     #-1.4210854715202005e-15, 0에 가까움.


#1. 정규성 : 잔차가 정규성을 따르는지 확인하기
from scipy.stats import shapiro
stat, pv = shapiro(residual)
print(f"shapiro-wilk test로 정규성 확인 : 통계량 : {stat:.4f}, p-value:{pv:.4f}")   #통계량 : 0.9905, p-value:0.2133
print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성이 있음')  #p값이 0.05보다 큼. 정규성 만족
import statsmodels.api as sm    # Quantile-Quantile plot(qq플롯 지원, 시각화로 확인)
sm.qqplot(residual, line='s')
plt.title('잔차 Q-Q plot')
plt.show()  #확인 결과, 정규성 만족이나, 커브를 그려나가는 부분이 좋지 않음.
plt.close()


#2. 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화하는지 확인하기
from statsmodels.stats.diagnostic import linear_reset   # 모형 적합성 확인
resetResult = linear_reset(lModel1, power=2, use_f=True)
print(f'linear_reset 테스트 : F={resetResult.fvalue:.4f}, p={resetResult.pvalue:.4f}')  #linear_reset 테스트 : F=3.7036, p=0.0557
print('선형성 만족' if resetResult.pvalue > 0.05 else '선형성 위배 가능성이 있음')  #선형성 만족, 그러나 간당간당 했음.
#시각화로 확인
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='gray')
plt.show()  #확인 결과, 잔차의 추세선(붉은선)이 파선(---)에 가까움, 선형성 충족함.
plt.close()


#3. 독립성 : 독립변수의 값이 서로 관련되지 않는지 확인하기.
#독립성 가정은 잔차 간에 자기상관이 없어야 함.
#자기상관 : 잔차 간 자기상관이란 회귀분석 등에서 계산된 잔차(예측값과 실제값의 차이)들이 서로 
#독립적이지 않고 시간 순서에 따라 연관성을 갖는 현상, 듀빈-왓슨(Durbin-Watson) 검정으로 확인 가능.
print(lModel1.summary())
#Durbin-Watson : 1.935, 2에 근사할수록 자기상관이 없음. 독립성 충족함.
#
#   |-------------2-------------|
#   0                           4
#2를 기준으로 작으면 음의 자기상관, 크면 양의 자기상관

#참고 : Cook's distance (쿡의 거리)
#하나의 관측치가 전체 모델에 얼마나 영향을 주는지 수치화한 지표.
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lModel1).cooks_distance
print(cd.sort_values(ascending = False).head(5))
#쿡값 중 가장 큰 다섯개만 보기 : 
#35     0.060494
#178    0.056347
#25     0.038873
#175    0.037181
#131    0.033895
print(advdf.iloc[[35, 178, 25, 175, 131]])
#인덱스 번째에 해당하는 원본 자료 확인 :
#        tv  radio  newspaper  sales
#35   290.7    4.1        8.5   12.8
#178  276.7    2.3       23.7   11.8
#25   262.9    3.5       19.5   12.0
#175  276.9   48.9       41.8   27.0
#131  265.2    2.9       43.0   12.7
#쿡값과 비교 결과, 대체적으로 TV 광고비는 높지만, 그에 비교해 Sales가 적음 - 모델이 예측하기 어려운 포인트들.
#쿡값 시각화
import statsmodels.api as sm
fig = sm.graphics.influence_plot(lModel1, alpha=0.05, criterion='cooks')
plt.show()
plt.close()