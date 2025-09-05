'''
판별 함수 기반 모형
분류 방법중 하나는 동일한 클래스가 모여 있는 영역과 그 영역을 나누는 경계면(boundary plane)을 정의하는 것이다. 
이 경계면은 경계면으로부터의 거리를 계산하는  f(x)형태의 함수인 판별 함수(discriminant function)로 정의된다. 
판별 함수의 값의 부호에 따라 클래스가 나뉘어진다.

Scikit-Learn 에서 분별 함수 기반의 모형은 분별 함수 값을 출력하는 decision_function 메서드를 제공한다.

퍼셉트론(Perceptron)은 가장 단순한 판별 함수 기반 모형이다. 
chart에서 직선이 경계선(boundary line)으로 데이터 영역을 나눈다.
'''
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
idx = np.in1d(iris.target, [0, 2])  # 1 차원 배열의 각 요소가 두 번째 배열에도 있는지 테스트
X = iris.data[idx, 0:2]
y = iris.target[idx]

model = Perceptron(max_iter=100, eta0=0.1, random_state=1).fit(X, y)
XX_min, XX_max = X[:, 0].min() - 1, X[:, 0].max() + 1
YY_min, YY_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

plt.contour(XX, YY, ZZ, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', linewidth=1)

idx = [22, 36, 70, 80]
plt.scatter(X[idx, 0], X[idx, 1], c='r', s=100, alpha=0.5)
for i in idx:
    plt.annotate(i, xy=(X[i, 0], X[i, 1] + 0.1))
plt.grid(False)
plt.show()

 
# bar chart
plt.bar(range(len(idx)), model.decision_function(X[idx]))
plt.xticks(range(len(idx)), idx)
plt.gca().xaxis.grid(False)
plt.title("Discriminant Function")
plt.show()

#만약 데이터의 차원이 3차원이라면 다음과 같이 경계면(boundary surface)을 가지게 된다. 
#이러한 경계면이나 경계선을 의사결정 하이퍼 플레인(decision hyperplane)이라고 한다.
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
idx = np.logical_or(iris.target == 0, iris.target == 1)
X = iris.data[idx, :3]
y = iris.target[idx]
fig = plt.figure()
ax = Axes3D(fig, elev=20, azim=10)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o', s=100);
ax.plot_surface(np.array([[4, 4], [7, 7]]), np.array([[2, 4.5], [2, 4.5]]),
                np.array([[2, 4], [2, 4]]), color='g', alpha=.3)
plt.show()
plt.close()