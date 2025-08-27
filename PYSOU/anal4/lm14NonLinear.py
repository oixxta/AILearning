"""
비선형회귀 분석
선형관계 분석의 경우, 모델에 다항식 또는 교호작용이 있는 경우에는, 해석이 덜 직관적임.(결과의 신뢰성이 떨어짐)
선형 과정이 어긋날 때(정규성 위배), 대처하는 방법으로 다항식 항을 추가한 다항회귀 모델을 작성할 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])

plt.scatter(x, y)
plt.show()
print(np.corrcoef(x, y))    #0.48076197

#선형회귀 모델로 만들어보기
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]    #리스트 전체의 차원 확대
model1 = LinearRegression().fit(x, y)
ypred = model1.predict(x)
print('예측값 : ', ypred)   #[2.  2.7 3.4 4.1 4.8]
                #실제값 :    [4   2   1   3   7  ]
print('결정계수1 : ', r2_score(y, ypred))       #0.23113207547169834
plt.scatter(x, y)
plt.plot(x, ypred, c='red')
plt.show()


#비선형회귀 모델(다항회귀 모델)로 만들어보기 - 추세선의 유연성을 위해 열 추가하기.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)  #degree : 열의 갯수, 원래의 데이터에 열 하나를 더 만들어서 행렬로 만듬.
x2 = poly.fit_transform(x)  # 특징 행렬을 만들기
print(x2)
model2 = LinearRegression().fit(x2, y)
ypred2 = model2.predict(x2)
print('예측값 : ', ypred2)   #[4.14285714 1.62857143 1.25714286 3.02857143 6.94285714]
                #실제값 :     [4          2          1          3          7         ]
print('결정계수1 : ', r2_score(y, ypred2))       #0.9892183288409704, 매우 적합함, 대신 오버피팅될 우려가 있음.
plt.scatter(x, y)
plt.plot(x, ypred2, c='blue')
plt.show()


#열 하나 더 추가하기
#from sklearn.preprocessing import PolynomialFeatures
#poly = PolynomialFeatures(degree=3, include_bias=False)
#x2 = poly.fit_transform(x)  # 특징 행렬을 만들기
#print(x2)
#model2 = LinearRegression().fit(x2, y)
#ypred2 = model2.predict(x2)
#print('예측값 : ', ypred2)   #[4.04285714 1.82857143 1.25714286 2.82857143 7.04285714]
#                #실제값 :     [4          2          1          3          7         ]
#print('결정계수1 : ', r2_score(y, ypred2))       #0.9939353099730458, 열 두개 였을때보다 결정계수가 더 커짐.
#plt.scatter(x, y)
#plt.plot(x, ypred2, c='blue')
#plt.show()
#확인 결과, 열이 더 추가될 경우, 모델의 성능이 크게 개선되나, 오버피팅될 우려가 있음!