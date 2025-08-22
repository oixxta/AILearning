"""
최소 제곱해를 선형 행렬 방정식으로 구하기
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.5, 2.1])
plt.scatter(x, y)
plt.show()
plt.close()

A = np.vstack([x, np.ones(len(x))]).T
print(A)

#위 데이터에 합리적인 추세선을 긋기 : 회귀분석
import numpy.linalg as lin
w, b = lin.lstsq(A, y, rcond=None)[0] # y = wx + b라는 일차 방정식에 쓸 w와 b 구하기, 최소제곱법 연산
#최소제곱법 : 잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있다.
print(w, b) # w : 0.9599999999999999 b : -0.9899999999999993, 각각 기울기와 절편
# 만든 수식 : y = 0.95999 * x + (-0.989999), 단순 성형회귀 수식 모델.

plt.scatter(x, y)
plt.plot(x, w * x + b, label='실제값')
plt.legend()
plt.show()
plt.close()

# 수식(w * x + b)을 사용해서 예측값 얻기
print(w * 1 + b)    #-0.02999999999999947, 실제 값은 0.2로, 예측값으로 얻은 값은 약간의 차이가 발생함.(잔차, 오차, 에러)
print(w * 0 + b)    # -0.9899999999999993, 실제 값은 -1
print(w * 2 + b)    # 0.9300000000000004, 살제 값은 0.5
print(w * 3 + b)    # 1.8900000000000001, 실제 값은 2.1