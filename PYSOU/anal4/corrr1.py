# 공분산과 상관계수
# 공분산 : 두 변수의 패턴을 확인하기 위해 공분산을 사용함. 단위 크기에 영향을 받음.
# 상관계수 : 공분산을 표준화(숫자화), -1 ~ 0 ~ 1 사이의 수치. +-1에 근사할 수록 관계가 강함.

import numpy as np
# 공분산 : 패턴 방향 확인 가능, 구체적인 크기를 표현하는 것은 곤란함.
print(np.cov(np.arange(1, 6), np.arange(2, 7)))
print(np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10)))
print(np.cov(np.arange(100, 600, 100), np.arange(200, 700, 100)))
print(np.cov(np.arange(1, 6), (3, 3, 3, 3, 3)))
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1)))
print('-----------------')

x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]
print('x 평균 : ', np.mean(x))  #x 평균 :  5.5
print('x 분산 : ', np.var(x))   #x 분산 :  5.45, 분산은 평균과의 거리와 관련이 있음.
y = [6, 2, 4, 6, 9, 5, 1, 8, 4, 5]
print('y 평균 : ', np.mean(y))  #y 평균 :  5.0
print('y 분산 : ', np.var(y))   #y 분산 :  5.4

import matplotlib.pyplot as plt
#plt.scatter(x, y)
#plt.show()
#plt.close()
print('x와 y의 공분산 : ', np.cov(x, y))
print('x와 y의 공분산 : ', np.cov(x, y)[0, 1])  #5.222222222222222, 우상향 패턴.

#공분산을 표준화 시키기(상관계수, -1 ~ 1 사이로)
print('x와 y의 상관계수 : ', np.corrcoef(x, y))
print('x와 y의 상관계수 : ', np.corrcoef(x, y)[0, 1])   #0.8663686463212855, 양의 상관관계 & 강한 관계(높은 밀집도)

#참고 : 비선형인 경우는 일반적인 상관계수를 사용할 수 없음.
m = [-3, -2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
#plt.scatter(m, n)
#plt.show()
#plt.close()
print(np.corrcoef(m, n)[0, 1])  #0.0, 무의미함.
#따라서, 상관계수를 확인하기 전에 산포도를 먼저 확인해서 선형인지 & 기울기가 0이 아닌지의 여부를 확인한 다음에 작업할 것!