"""
학생 10명의 시험 점수로 KMeans 수행해보기 (비 계층적 군집 분류 : k-means로)

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

students = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1, 1)
print('점수 : ', scores.ravel())    #점수 :  [76 95 65 85 60 92 55 88 83 72]

# 학생 10명의 시험 점수로 Kmeans 수행, k 는 3으로 지정.
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_clust = kmeans.fit_predict(scores)

df = pd.DataFrame({
    'Student' : students,
    'Score' : scores.ravel(),
    'Cluster' : kmeans_clust
})

# 군집결과
print('군집 결과 : \n', df)
"""
군집 결과 : 
   Student  Score  Cluster
0      s1     76        2
1      s2     95        0
2      s3     65        1
3      s4     85        2
4      s5     60        1
5      s6     92        0
6      s7     55        1
7      s8     88        0
8      s9     83        2
9     s10     72        2
"""

# 군집별 평균 점수
print('군집별 평균 점수 : ')
grouped = df.groupby('Cluster')['Score'].mean()
print(grouped)
"""
군집별 평균 점수 :
Cluster
0    91.666667
1    60.000000
2    79.000000
"""

# 시각화
xPositions = np.arange(len(students))
yScores = scores.ravel()
colors = {0: 'red', 1: 'blue', 2: 'black'}
plt.figure(figsize=(10, 6))
for i, (x, y, cluster) in enumerate(zip(xPositions, yScores, kmeans_clust)):
    plt.scatter(x, y, color=colors[cluster], s=100)
    plt.text(x, y + 1.5, students[i], fontsize=10, ha='center')


# 중심점 표시
centers = kmeans.cluster_centers_
for center in centers:
    plt.scatter(len(students) // 2, center[0], marker='X', c='gold', s=200)
plt.xticks(xPositions, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.title('K-means Clustering for Students Score')
plt.grid()
plt.show()
plt.close()

