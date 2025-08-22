# lm 3에서 이어짐

# 방법 4 : linregress   model0
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측하기 : IQ가 시험점수에 영향을 주는가?
#데이터 가져오기
score_iq = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv")
print(score_iq)
print(score_iq.info())

#상관관계 확인하기(상관계수를 확인해서)
x = score_iq.iq
y = score_iq.score
print(np.corrcoef(x, y))    #상관계수 : 0.88222034. 두 개는 서로 강한 상관관계를 가짐. 그래프상 우상향.
print(score_iq.corr())

plt.scatter(x, y)
plt.show()
plt.close()
#두 데이터가 상관관계를 갖고 있기에, 회귀분석으로 넘어갈 수 있음.


#회귀분석 하기
model = stats.linregress(x=x, y=y)
print(model)    #slope=np.float64(0.6514309527270075), intercept=np.float64(-2.8564471221974657), rvalue=np.float64(0.8822203446134699), pvalue=np.float64(2.8476895206683644e-50)
#기울기 : 0.6514, 절편 : -2.856, 결정계수 : 0.8822, p값 : 0에 가까움, 이 모델은 두 변수간의 인과관계가 있고, 의미있는 모델임을 의미.
print('slope : ', model.slope)
print('intercept : ', model.intercept)
print('R² : ', model.rvalue)    #결정계수 : 독립변수가 종속변수를 88%정도 설명하고 있음.
print('pvalue : ', model.pvalue)    # p값이 0.05보다 작음으로 현재 모델은 유의함을 의미함. (독립변수와 종속변수는 인과관계가 있다.)
print('표준오차 : ', model.stderr)  #
#위 회귀분석 결과로 만든 수식 : y = 0.6514 * x + (-2.856)
plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept)
plt.show()
plt.close()


#점수예측
print('iq가 80일때의 점수 예측 : ', model.slope * 80 + model.intercept) #49.25802909596313
print(np.polyval([model.slope ,model.intercept], np.array(score_iq['iq'][:5]))) #[88.34388626 78.57242197 75.31526721 85.0867315  65.54380291]

newDf = pd.DataFrame({'iq': [55, 66, 77, 88, 150]})
print(np.polyval([model.slope ,model.intercept], newDf))
"""
[[32.97225528]
 [40.13799576]
 [47.30373624]
 [54.46947672]
 [94.85819579]]
"""