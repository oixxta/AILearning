"""
날씨예보(강우 여부)

독립변수 : RainTomorrow를 제외한 나머지
종속변수 : RainTomorrow
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

#데이터 가져오기
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(data.head(3), data.shape)  #(366, 12)
data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis=1)    #쓸 데이터만 넣기
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})    #범주형 칼럼을 0과 1로 변환
print(data2.head(3), data.shape) #(366, 12)
"""
   MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp  RainTomorrow
0      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6             1
1     14.0     26.9       3.6       9.7         17        36    1008.4      3  25.7             1
2     13.7     23.4       3.6       3.3          6        69    1007.2      7  20.2             1
"""
print(data2.RainTomorrow.unique())  #RainTomorrow가 모두 0과 1만 있는지 재확인


#학습용 데이터와 검정용 데이터로 나누기
trainData, testData = train_test_split(data2, test_size=0.3, random_state=42)
print(trainData.shape, testData.shape)  #(256, 10) (110, 10)
print(data2.columns)
columnsSelector = "+".join(trainData.columns.difference(['RainTomorrow']))  #RainTomorrow를 제외한 나머지 칼럼들을 저장한 스트링문
print(columnsSelector)  # Cloud+Humidity+MaxTemp+MinTemp+Pressure+Rainfall+Sunshine+Temp+WindSpeed
myFormula = 'RainTomorrow ~ ' + columnsSelector

#model = smf.glm(formula = myFormula, data = trainData, family = sm.families.Binomial()).fit()
model = smf.logit(formula = myFormula, data = trainData).fit()

print(model.summary())
"""
                 Generalized Linear Model Regression Results
==============================================================================
Dep. Variable:           RainTomorrow   No. Observations:                  253
Model:                            GLM   Df Residuals:                      243
Model Family:                Binomial   Df Model:                            9
Link Function:                  Logit   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -72.927
Date:                Thu, 28 Aug 2025   Deviance:                       145.85
Time:                        11:56:00   Pearson chi2:                     194.
No. Iterations:                     6   Pseudo R-squ. (CS):             0.3186
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    219.3889     53.366      4.111      0.000     114.794     323.984
Cloud          0.0616      0.118      0.523      0.601      -0.169       0.293
Humidity       0.0554      0.028      1.966      0.049       0.000       0.111
MaxTemp        0.1746      0.269      0.649      0.516      -0.353       0.702
MinTemp       -0.1360      0.077     -1.758      0.079      -0.288       0.016
Pressure      -0.2216      0.052     -4.276      0.000      -0.323      -0.120
Rainfall      -0.1362      0.078     -1.737      0.082      -0.290       0.018
Sunshine      -0.3197      0.117     -2.727      0.006      -0.550      -0.090
Temp           0.0428      0.272      0.157      0.875      -0.489       0.575
WindSpeed      0.0038      0.032      0.119      0.906      -0.059       0.066
==============================================================================
"""
print(model.params)
"""
Intercept    219.388868
Cloud          0.061599
Humidity       0.055433
MaxTemp        0.174591
MinTemp       -0.136011
Pressure      -0.221634
Rainfall      -0.136161
Sunshine      -0.319738
Temp           0.042755
WindSpeed      0.003785
"""
print('예측값 : ', np.rint(model.predict(testData)[:5]))    #[0, 0, 0, 0, 0]
print('실제값 : ', testData['RainTomorrow'][:5])            #[0, 0, 0, 0 ,0], 100%일치.


#분류 정확도 확인하기
conf_table = model.pred_table()
print('conf_tab : \n', conf_table)     #GLM은 logit이랑 달리 PRED TABLE을 지원하지 않음!
"""
conf_tab :
 [[197.   9.]
 [ 21.  26.]]   # 모델이 맞춘 갯수 : 197 + 26 = 
"""
print('분류 정확도 : ', ((conf_table[0][0] + conf_table[1][1]) / len(trainData)))   #분류 정확도 :  0.87109375

from sklearn.metrics import accuracy_score
pred = model.predict(testData)
print('분류 정확도 : ', accuracy_score(testData['RainTomorrow'], np.around(pred)))  #분류 정확도 :  0.8727272727272727, 87.3%의 정확도로 간주
