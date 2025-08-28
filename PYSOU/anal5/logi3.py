"""
Logistic Regression 클래스 : 다항 분류가 가능함.
활성화 함수는 softmax를 사용함.
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()

#print(iris.DESCR)
print(iris.keys())
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
print(iris.target)

x = iris['data'][:, [3]]    # petal.length
#print(x)
y = (iris.target == 2).astype(np.int32)
#print(y[:3])
#print(type(y))

log_reg = LogisticRegression().fit(x, y)     # solver : lbfgs (softmax 사용)
print(log_reg)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)  #새로운 예측값을 얻기 위해 독립변수 생성하기
#print(x_new)
y_prob = log_reg.predict_proba(x_new)
#print(y_prob)

import matplotlib.pyplot as plt
plt.plot(x_new, y_prob[:, 1], 'r-', label='virginica')
plt.plot(x_new, y_prob[:, 0], 'b--', label='not virginica')
plt.legend()
plt.xlabel('petal width')
plt.show()
plt.close()
