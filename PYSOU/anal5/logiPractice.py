"""
[로지스틱 분류분석 문제1]

문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 

다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.

키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

독립변수 : 소득수준
종속변수 : 외식여부
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

#데이터 가져오기
originData = pd.read_csv('dinner.csv')
print(originData.head(3))
"""
  요일  외식유무  소득수준
0  토     0    57
1  토     0    39
2  토     0    28
"""
data = pd.DataFrame()
data = originData.drop(['요일'], axis=1)
print(data.head(3))

#학습용 데이터와 검정용 데이터로 나누기(데이터가 적어서 생략해도 되지만, 그냥 함.)
trainData, testData = train_test_split(data, test_size=0.3, random_state=1)
print(trainData.shape, testData.shape)  #(19, 2) (9, 2)


#데이터 학습하기 : GLM과 LOGIT 두개의 모델
modelGlm = smf.glm(formula='외식유무 ~ 소득수준', data=trainData, family = sm.families.Binomial()).fit()
print('예측값 : ', np.rint(modelGlm.predict(testData)[:5]))     #[1, 0, 0, 1, 1]
print('실제값 : ', testData['외식유무'][:5])                     #[1, 0, 1, 1, 1], 1개 틀림

modelLogit = smf.logit(formula='외식유무 ~ 소득수준', data=trainData).fit()
print(modelLogit.summary())
"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                외식유무   No. Observations:                   19
Model:                          Logit   Df Residuals:                       17
Method:                           MLE   Df Model:                            1
Date:                Thu, 28 Aug 2025   Pseudo R-squ.:                  0.3777
Time:                        12:47:30   Log-Likelihood:                -8.0473
converged:                       True   LL-Null:                       -12.932
Covariance Type:            nonrobust   LLR p-value:                  0.001774
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -8.6151      3.819     -2.256      0.024     -16.100      -1.130
소득수준        0.1823      0.084      2.182      0.029       0.019       0.346
==============================================================================
"""
print(modelLogit.params)
"""
Intercept      -8.615104
소득수준         0.182276
"""
print('예측값 : ', np.rint(modelLogit.predict(testData)[:5]))   #[1, 0, 0, 1, 1]
print('실제값 : ', testData['외식유무'][:5])                     #[1, 0, 1, 1, 1]


#분류 정확도 확인하기
confTable = modelLogit.pred_table()
print(confTable)        #GLM은 logit이랑 달리 PRED TABLE을 지원하지 않음!
#[[10.  1.]
# [ 3.  5.]]            trainData에 있는 것들 중 모델이 맞춘 갯수 : 참음성과(10) + 참양성(5) = 15
#전체 19개 중 15개 정답
print('분류 정확도 by confusion matrix : ', (confTable[0][0] + confTable[1][1]) / len(trainData))
#0.7894736842105263, 분류 정확도는 78.9%

from sklearn.metrics import accuracy_score
pred = modelGlm.predict(testData)
print('분류 정확도 by accuracy_score : ', accuracy_score(testData['외식유무'], np.around(pred)))
#0.8888888888888888, 분류 정확도는 88.8%