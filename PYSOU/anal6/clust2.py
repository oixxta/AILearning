"""
계층적 군집
10명의 학생의 시험 점수를 사용

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

students = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
scores = np.array([76, 95, 65, 85, 60, 92, 55, 88, 83, 72]).reshape(-1, 1)
print('점수 : ', scores.ravel())    #점수 :  [76 95 65 85 60 92 55 88 83 72]

#계층적 군집
linked = linkage(scores, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, labels=students)
plt.axhline(y=25, color='red', linestyle='--', label='cut at height=25')
plt.xlabel('Students')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.show()
plt.close()
#점수 상위그룹과 하위그룹 두 개로 잘 나뉘어짐.


#군집 3개로 나누기
clusters = fcluster(linked, 3, criterion='maxclust')
print(clusters) #[2 1 3 1 3 1 3 1 1 2]
for student, cluster in zip(students, clusters):
    print(f'{student} : cluster {cluster}')
"""
s1 : cluster 2
s2 : cluster 1
s3 : cluster 3
s4 : cluster 1
s5 : cluster 3
s6 : cluster 1
s7 : cluster 3
s8 : cluster 1
s9 : cluster 1
s10 : cluster 2

3개의 군집으로 나뉘어짐
"""
#군집별로 점수와 이름 정리
cluster_info = {}
for student, cluster, score in zip(students, clusters, scores.ravel()):
    if cluster not in cluster_info:
        cluster_info[cluster] = {'students': [], 'scores': []}
    cluster_info[cluster]['students'].append(student)
    cluster_info[cluster]['scores'].append(score)

for cluster_id, info in sorted(cluster_info.items()):
    avg_score = np.mean(info['scores'])
    student_list = ', '.join(info['students'])
    print(f'cluster {cluster_id} : 평균점수 {avg_score:.2f}, 학생들 {student_list}')
"""
cluster 1 : 평균점수 88.60, 학생들 s2, s4, s6, s8, s9
cluster 2 : 평균점수 74.00, 학생들 s1, s10
cluster 3 : 평균점수 60.00, 학생들 s3, s5, s7
"""
#군집별로 시각화 하기
xPositions = np.arange(len(students))
yScore = scores.ravel()
colors = {1:'red', 2:'blue', 3:'green'}
plt.figure(figsize=(10, 6))

for i, (x, y, cluster) in enumerate(zip(xPositions, yScore, clusters)):
    plt.scatter(x, y, color=colors[cluster], s=100)
    plt.text(x, y + 1.5, students[i], fontsize=10, ha='center')
plt.xticks(xPositions, students)
plt.xlabel('Students')
plt.ylabel('Score')
plt.grid()
plt.show()
plt.close()