# 계층적 군집 - Iris dataset 사용
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

iris = load_iris()
X = iris.data   
y = iris.target                    
target_names = iris.target_names   # ['setosa', 'versicolor', 'virginica']

# 라벨(학생 이름 대신 샘플 ID)
labels = [f's{i+1}' for i in range(len(X))]

# 표준화 (Ward는 스케일 영향 큼)
X_scaled = StandardScaler().fit_transform(X)

# 계층적 군집 (Ward)
Z = linkage(X_scaled, method='ward')

# 덴드로그램 그리기
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=labels, leaf_font_size=6)
# "3개 군집" 높이 기준선을 자동 계산: Z[-3]은 3개 군집이 막 형성된 시점, Z[-2]는 2개로 줄어드는 시점
cut_height = (Z[-3, 2] + Z[-2, 2]) / 2
plt.axhline(y=cut_height, color='red', linestyle='--', label=f'cut @ {cut_height:.2f}')
plt.xlabel('Samples')
plt.ylabel('Distance (Ward)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# fcluster로 평면 군집 라벨 얻기 (3개 군집)
clusters = fcluster(Z, 3, criterion='maxclust')  # 값은 1,2,3 중 하나
print("클러스터 라벨(앞 20개):", clusters[:20])

# 군집별 요약 (샘플 수 / 예시 이름 몇 개)
from collections import defaultdict, Counter
cluster_members = defaultdict(list)
for lab, c in zip(labels, clusters):
    cluster_members[c].append(lab)

print("\n군집 요약:")
for cid in sorted(cluster_members):
    members = cluster_members[cid]
    print(f"- Cluster {cid}: {len(members)}개  예시:", ", ".join(members[:10]), "...")

# 실제 품종과의 매핑 대략 보기
print("\n(참고) 실제 품종 분포 vs 군집")
for cid in sorted(set(clusters)):
    idx = np.where(clusters == cid)[0]
    counts = Counter(y[idx])
    pretty = {target_names[k]: v for k, v in counts.items()}
    print(f"  Cluster {cid}: {pretty}")

# 시각화: PCA 2D로 투영해 군집 색상 표시
# 4차원이라 점 그래프는 PCA로 2차원 투영해서 색으로 군집을 표현
pca = PCA(n_components=2, random_state=0)
X_2d = pca.fit_transform(X_scaled)

colors = {1: 'tab:red', 2: 'tab:blue', 3: 'tab:green'}
plt.figure(figsize=(8, 6))
for cid in sorted(set(clusters)):
    idx = np.where(clusters == cid)[0]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=60, alpha=0.8, label=f'Cluster {cid}', c=colors[cid])

plt.title('Iris / Ward 계층적 군집 (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
