"""
아이리스 데이터를 가져와서 계층적 군집 분석하기
"""
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head(3))
"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
"""
print(iris_df.loc[0:2, ['sepal length (cm)', 'sepal width (cm)']])
"""
   sepal length (cm)  sepal width (cm)
0                5.1               3.5
1                4.9               3.0
2                4.7               3.2
"""
from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')
print('dist_vec : ', dist_vec)
#dist_vec :  [0.53851648 0.5        0.64031242 0.14142136 0.28284271 0.31622777
# 0.60827625 0.14142136 0.5        0.64031242]
print()
row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)
"""
          0         1         2         3         4
0  0.000000  0.538516  0.500000  0.640312  0.141421
1  0.538516  0.000000  0.282843  0.316228  0.608276
2  0.500000  0.282843  0.000000  0.141421  0.500000
3  0.640312  0.316228  0.141421  0.000000  0.640312
4  0.141421  0.608276  0.500000  0.640312  0.000000
"""
from scipy.cluster.hierarchy import linkage, dendrogram
row_clusters = linkage(dist_vec, method='complete') #완전연결법 사용
print('row_clusters : ', row_clusters)
df = pd.DataFrame(row_clusters, columns=['id1', 'id2', '거리', '맴버수'])
print(df)
"""
   id1  id2       거리  맴버수
0  0.0  4.0  0.141421    2.0
1  2.0  3.0  0.141421    2.0
2  1.0  6.0  0.316228    3.0
3  5.0  7.0  0.640312    5.0
"""
row_dend = dendrogram(row_clusters)
plt.tight_layout()
plt.ylabel('dist')
plt.show()

print()
from sklearn.cluster import AgglomerativeClustering #클러스터 정보를 확인할 때 사용.
ac = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')
X = iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']]
labels = ac.fit_predict(X)
print('클러스터 분류 결과 : ', labels)      #클러스터 분류 결과 :  [1 0 0 0 1]

plt.hist(labels)
plt.grid()
plt.show()
plt.close()
