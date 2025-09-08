"""
클러스터링 기법 중 계층적 군집화 이해하기

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')

np.random.seed(123)
var = ['X', 'Y']
labels = ['점0', '점1', '점2', '점3', '점4']
X = np.random.random_sample([5, 2]) * 10 #5행 2열짜리 랜덤샘플 생성 후 0 에서 9까지의 실수 반환
df = pd.DataFrame(X, columns=var, index=labels)
print(df)
"""
           X         Y
점0  6.964692  2.861393
점1  2.268515  5.513148
점2  7.194690  4.231065
점3  9.807642  6.848297
점4  4.809319  3.921175
"""

# plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', s=50)
# plt.grid(True)
# plt.show()

from scipy.spatial.distance import pdist, squareform   #클러스터링 구경을 위해 호출, 요소들의 거리를 계산함.
#pdist : 배열에 있는 값을 이용해 각 요소들의 거리를 계산.
#squareform : 거리벡터를 사각형 형식으로 변환하는 역할.
distVec = pdist(df, metric='euclidean')             #(데이터, metric=계산방법), 유클리디안 계산법이 가장 일반적임.
print('distVec : ', distVec)
#distVec :  [5.3931329  1.38884785 4.89671004 2.40182631 5.09027885 7.6564396
#  2.99834352 3.69830057 2.40541571 5.79234641], 거리 계산을 했음. 그러나, 뭐가 뭐와의 거리인지 알 수 없음.

rowDist = pd.DataFrame(squareform(distVec), columns=labels, index=labels)
print(rowDist)
"""
          점0        점1       점2       점3        점4
점0  0.000000  5.393133  1.388848  4.896710  2.401826
점1  5.393133  0.000000  5.090279  7.656440  2.998344
점2  1.388848  5.090279  0.000000  3.698301  2.405416
점3  4.896710  7.656440  3.698301  0.000000  5.792346
점4  2.401826  2.998344  2.405416  5.792346  0.000000

위의 distVec을 보기 좋게 표로 만듬.
"""

# 응집형 : 자료 하나하나를 군집으로 보고, 가까운 군집끼리 연결해 가는 방법. 상향식.
# 분리형 : 전체 자료를 하나의 군집으로 보고 분리해 나가는 방법. 하향식.

# linkage : 응집형, 계층적 군집을 수행.
from scipy.cluster.hierarchy import linkage
rowClusters = linkage(distVec, method='ward')   #와드연결법을 사용함.
df = pd.DataFrame(rowClusters, columns=['클러스터 id 1', '클러스터 id 2', '거리', '클러스터 맴버 수'])
print(df)
"""
   클러스터 id 1  클러스터 id 2        거리  클러스터 맴버 수
0           0.0          2.0    1.388848            2.0
1           4.0          5.0    2.657109            3.0     #점 0과 2를 합친게 점5
2           1.0          6.0    5.454004            4.0     #점 4과 5를 합친게 점6
3           3.0          7.0    6.647102            5.0     #점 6과 1를 합친게 점7
"""
# linkage의 결과를 시각화 하기 : 덴드로그램 작성
from scipy.cluster.hierarchy import dendrogram
rowDendr = dendrogram(rowClusters, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()
plt.close()
