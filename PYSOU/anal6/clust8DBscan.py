"""
DBSCAN (밀도기반 클러스터링 알고리즘)

밀도 기반 클러스터링 비모수적 알고리즘이다. 일부 공간에 있는 점의 경우, 
서로 밀접하게 밀집된 점(인근 이웃이 많은 점)을 그룹화하여 저밀도 지역
(가장 가까운 이웃이 너무 멀리 떨어져 있음)에 혼자 있는 이상점으로 표시한다.

K-means와 DBSCAN의 비교:

"""
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

x, y = make_moons(n_samples=200, noise=0.05, random_state=0)
print(x[:10])
print('실제 군집 id : ', y[:10])
"""
[[ 0.81680544  0.5216447 ]
 [ 1.61859642 -0.37982927]
 [-0.02126953  0.27372826]
 [-1.02181041 -0.07543984]
 [ 1.76654633 -0.17069874]
 [ 1.8820287  -0.04238449]
 [ 0.97481551  0.20999374]
 [ 0.88798782 -0.48936735]
 [ 0.89865156  0.36637762]
 [ 1.11638974 -0.53460385]]
실제 군집 id :  [0 1 1 0 1 1 0 1 0 1]
"""
# 데이터 분포 확인 : 밀도기반 클러스터링이 아니면 구분 불가!
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

#KMeans로 군집화 해보기
km = KMeans(n_clusters=2, random_state=0)
pred1 = km.fit_predict(x)
print('예측 군집 id : ', pred1[:10])
#실제 군집 id :  [0 1 1 0 1 1 0 1 0 1]
#예측 군집 id :  [1 1 0 0 1 1 1 1 1 1]
#시각화 해보기
def plotResultFunc(x, pr):
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='cluster-1')
    plt.scatter(x[pr==1, 0], x[pr==1, 1], c='red', marker='s', s=40, label='cluster-2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='black', marker='+', s=50, label='centroid')
    plt.legend()
    plt.title('dsdsd')
    plt.show()
    plt.close()

plotResultFunc(x, pred1)        #이미지상, 제대로 분류되지 못했음.


#DBSCAN으로 군집화 해보기
dm = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')    
#eps : 두 샘플간 최대거리, min_samples : 반경 포인트들의 최소 갯수
pred2 = dm.fit_predict(x)

plotResultFunc(x, pred2)        #이번엔 제대로 분류되었음.
#군집화 : 고객 세분화, 예상치 탐지, 추천시스템 등에 효과적임.
