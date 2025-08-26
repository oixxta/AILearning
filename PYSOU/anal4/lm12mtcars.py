#리니어 리그래션을 사용한 선형회귀 연습

import numpy as np
import pandas as pd
import statsmodels.api
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

#데이터 긁어오기
mtCars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtCars.head(3))
"""
                mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
rownames
Mazda RX4      21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
Datsun 710     22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
"""

#상관계수 확인
print(mtCars.corr(method='pearson'))
"""
           mpg       cyl      disp        hp      drat        wt      qsec        vs        am      gear      carb
mpg   1.000000 -0.852162 -0.847551 -0.776168  0.681172 -0.867659  0.418684  0.664039  0.599832  0.480285 -0.550925
cyl  -0.852162  1.000000  0.902033  0.832447 -0.699938  0.782496 -0.591242 -0.810812 -0.522607 -0.492687  0.526988
disp -0.847551  0.902033  1.000000  0.790949 -0.710214  0.887980 -0.433698 -0.710416 -0.591227 -0.555569  0.394977
hp   -0.776168  0.832447  0.790949  1.000000 -0.448759  0.658748 -0.708223 -0.723097 -0.243204 -0.125704  0.749812
drat  0.681172 -0.699938 -0.710214 -0.448759  1.000000 -0.712441  0.091205  0.440278  0.712711  0.699610 -0.090790
wt   -0.867659  0.782496  0.887980  0.658748 -0.712441  1.000000 -0.174716 -0.554916 -0.692495 -0.583287  0.427606
qsec  0.418684 -0.591242 -0.433698 -0.708223  0.091205 -0.174716  1.000000  0.744535 -0.229861 -0.212682 -0.656249
vs    0.664039 -0.810812 -0.710416 -0.723097  0.440278 -0.554916  0.744535  1.000000  0.168345  0.206023 -0.569607
am    0.599832 -0.522607 -0.591227 -0.243204  0.712711 -0.692495 -0.229861  0.168345  1.000000  0.794059  0.057534
gear  0.480285 -0.492687 -0.555569 -0.125704  0.699610 -0.583287 -0.212682  0.206023  0.794059  1.000000  0.274073
carb -0.550925  0.526988  0.394977  0.749812 -0.090790  0.427606 -0.656249 -0.569607  0.057534  0.274073  1.000000
"""

#mpg를 종속변수로, hp를 독립변수로 하는 모델 만들기
# sklearn모듈은 주로 입력은 2차원, 출력은 1차원으로 만들어짐.
x = mtCars[['hp']].values
print(x[:3])
y = mtCars['mpg'].values
print(y[:3])

lModel = LinearRegression().fit(x, y)
print('slope : ', lModel.coef_)                 #기울기 : -0.06822828
print('intercept : ', lModel.intercept_)        #절편 : 30.098860539622496

plt.scatter(x, y)
plt.plot(x, lModel.coef_ * x + lModel.intercept_, c='r')
plt.show()

#모델 테스트하기
pred = lModel.predict(x)
print('예측값 : ', np.round(pred[:5], 1))       #[22.6 22.6 23.8 22.6 18.2]
print('실제값 : ', y[:5])                       #[21.  21.  22.8 21.4 18.7]
print()

#모델 성능평가
print('MSE : ', mean_absolute_error(y, pred))   #2.907452474234763
print('r2_score : ', r2_score(y, pred))         #0.602437341423934

#새로운 hp에 대한 mpg 구해보기
newHp = [[123]]
newPred = lModel.predict(newHp)
print('%s 마력인 경우 연비는 약 %s입니다'%(newHp[0][0], newPred))   #123 마력인 경우 연비는 약 [21.70678234]입니다