"""
Logistic Regression

독립(feature, x) : 연속형, 종속변수(label, y) : 범주형
이항 분류(다항 분류도 가능은 함.)
출력된 연속형(확률)자료를 logit 변환해 최종적으론 sigmoid function에 의해 0에서부터 1사이의 
실수값이 나오는데, 0.5를 기준으로 0과 1로 분류함.


sigmoid function 맛보기
"""
import math

def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

print(sigmoidFunc(3))       #0.9525741268224334
print(sigmoidFunc(1))       #0.7310585786300049
print(sigmoidFunc(-123))    #3.817497188671175e-54
print(sigmoidFunc(0.123))   #0.5307112905000478

# mtcar dataset 사용
import statsmodels.api as sm
mtcarData = sm.datasets.get_rdataset('mtcars')
print(mtcarData.keys())     #dict_keys(['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data'])
mtcars = mtcarData.data
print(mtcars.head(2))
"""
                mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
rownames
Mazda RX4      21.0    6  160.0  110   3.9  2.620  16.46   0   1     4     4
Mazda RX4 Wag  21.0    6  160.0  110   3.9  2.875  17.02   0   1     4     4
"""
# mpg(연비), hp(마력)가 am(자동)에 영향을 준다고 한다 : 독립변수 2개, 종속변수 1개
mtcar = mtcars.loc[:, ['mpg', 'hp', 'am']]
print(mtcar.head(2))
"""
                mpg   hp  am
rownames
Mazda RX4      21.0  110   1
Mazda RX4 Wag  21.0  110   1
"""
print(mtcar['am'].unique())

#연비와 마력수에 따른 변속기 분류 모델 작성(수동, 자동)
#모델 작성 방법1 : logit()
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'
model1 = smf.logit(formula=formula, data=mtcar).fit()
print(model1.summary()) #Logit Regression 결과표 출력
"""
                           Logit Regression Results
==============================================================================
Dep. Variable:                     am   No. Observations:                   32
Model:                          Logit   Df Residuals:                       29
Method:                           MLE   Df Model:                            2
Date:                Wed, 27 Aug 2025   Pseudo R-squ.:                  0.5551
Time:                        17:08:56   Log-Likelihood:                -9.6163
converged:                       True   LL-Null:                       -21.615
Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
hp             0.0550      0.027      2.045      0.041       0.002       0.108
mpg            1.2596      0.567      2.220      0.026       0.147       2.372
==============================================================================

결과표에는 모델이 유의한지의 여부에 대해선 나오지 않음, 

"""
#예측값 / 실제값 출력하기
import numpy as np
print('예측값 : ', model1.predict())
pred = model1.predict(mtcar[:10])
print('예측값 : ', pred.values)
#[0.25004729 0.25004729 0.55803435 0.35559974 0.39709691 0.00651918 0.10844152 0.63232168 0.58498645 0.06598365]
print('예측값 : ', np.around(pred.values))
print('실제값 : ', mtcar['am'][:10].values)
"""
예측값 :  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
실제값 :  [1  1  1  0  0  0  0  0  0  0]
"""
print()

#분류 모델의 정확도(accuracy) 확인
conf_tab = model1.pred_table()  # 수치에 대한 집계표 제공
print('confusion matrix : \n', conf_tab)
"""
confusion matrix :
 [[16.  3.]             #실제값
 [ 3. 10.]]             #예측값
"""
print('분류 정확도 : ', (16 + 10) / len(mtcar))     #모델이 맞춘 갯수 / 전체 갯수
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))
"""
분류 정확도 :  0.8125       #81.25퍼센트 정확도의 분류 모델
분류 정확도 :  0.8125       #81.25퍼센트 정확도의 분류 모델
"""