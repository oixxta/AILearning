"""
가상의 데이터로 쇼핑몰 고객 세분화 하기.(집단화) 연습.

비지도학습(K-Means, DBSCAN 비교)
칼럼 : 고객 수, 연간지출액, 방문횟수 등...
DBSCAN 사용

DBSCAN 시에는 데이터에 대한 표준화(StandardScaler)가 추천됨 : 이상치(노이즈) 제거 효과도 있음.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
n_customers = 200   # 고객 수
annual_spending = np.random.normal(50000, 15000, n_customers)  
#연간지출액. 표준편차를 따르는 무작위 수
monthly_visits = np.random.normal(5, 2, n_customers)  #월 방문 횟수
print(annual_spending[:5])
print(monthly_visits[:5])

#    numpy.clip(array, min, max) : 수치 안정화 함수. 범위 고정, 넘파이 제공
#    array 내의 element들에 대해서
#    min 값 보다 작은 값들을 min값으로 바꿔주고
#    max 값 보다 큰 값들을 max값으로 바꿔주는 함수.

annual_spending = np.clip(annual_spending, 0, None)
monthly_visits = np.clip(monthly_visits, 0, None)
print(annual_spending[:5])
print(monthly_visits[:5])

data = pd.DataFrame({
    'annual_spending' : annual_spending,
    'monthly_visits' : monthly_visits
})
print(data.head(5))
"""
   annual_spending  monthly_visits
0     76460.785190        4.261636
1     56002.358126        4.521242
2     64681.069762        7.199319
3     83613.397988        6.310527
4     78013.369852        6.280263
"""

#산포도 그려보기
plt.scatter(data['annual_spending'], data['monthly_visits'])
plt.xlabel('annual_spending')
plt.ylabel('monthly_visits')
plt.show()
plt.close()

#표준화(StandardScaler) 실시.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print(data_scaled[:2])

#군집화 실시 : DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(data_scaled)
data['cluster'] = clusters
print(data.head(3))

#시각화
for cluster_id in np.unique(clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    plt.scatter(cluster_data['annual_spending'], cluster_data['monthly_visits'], label=f'cluster {cluster_id}')
plt.xlabel('annual_spending')
plt.ylabel('monthly_visits')
plt.legend()
plt.show()
plt.close()

print(data['cluster'].value_counts())
"""
cluster
 0    126       #메인 군집
-1     64       #노이즈 : 양 쪽 군집에 끼지 못함.
 1     10       #소규모 군집(독특한 패턴을 가진 고객)
"""