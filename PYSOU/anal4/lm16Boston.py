"""
보스턴 집값 데이터를 활용한 비선형회귀 연습
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#data = pd.read_csv("house_price.csv")
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data", header=None, sep=r'\s+')
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(data.head(3))

print(data.corr())    #상관계수 확인하기, MEDV / LSTAT : -0.737663, 음의 상관관계
x = data[['LSTAT']].values  #하위계층 비율
y = data['MEDV'].values     #지역의 집값 중앙값

model = LinearRegression()
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)

#단순회귀
model.fit(x, y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)
print(y_lin_fit)
model_r2 = r2_score(y, model.predict(x))
print('model_r2 : ', model_r2)              #0.5441462975864799

#다항회귀 (2차)
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
q_r2 = r2_score(y, model.predict(x_quad))
print('quad_r2 : ', q_r2)                   #0.6407168971636611

#다항회귀 (3차)
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, model.predict(x_cubic))
print('quad_c2 : ', c_r2)                   #0.6578476405895719

#시각화
plt.scatter(x, y, label='Training data', c = 'lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1),$R^2=%.2f$'%model_r2, c='b', lw=3) #설명력 : 54%
plt.plot(x_fit, y_quad_fit, linestyle='-', label='linear fit(d=2),$R^2=%.2f$'%q_r2, c='r', lw=3)    #설명력 : 64%
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='linear fit(d=3),$R^2=%.2f$'%c_r2, c='g', lw=3)  #설명력 : 65%
plt.xlabel('lower Class ratio')
plt.ylabel('house pay')
plt.legend()
plt.show()



