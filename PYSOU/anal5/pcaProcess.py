import matplotlib.pyplot as plt
import numpy as np

# 2차원 샘플 데이터 생성
np.random.seed(42)
x = np.random.normal(0, 1, 100)
# y는 x에 비례하지만 약간씩 흩어진 데이터 생성
y = 2*x + np.random.normal(0, 0.5, 100)
X = np.vstack([x, y]).T   # x와 y를 위아래로 쌓은 후 (2 × 100 형태의 행렬)  [[x0, x1, ..., x99], [y0, y1, ..., y99]]

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title("original data")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.axis("equal")
plt.grid(True)
plt.show()

# 평균 중심화된 데이터 시각화 
X_centered = X - X.mean(axis=0)

plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Centralized data")
plt.axis("equal")
plt.grid(True)
plt.show()

C = np.cov(X_centered.T)   # 공분산 행렬
print("공분산 행렬:\n", C)

# 공분산 행렬을 고유값 분해(Eigendecomposition) 하는 부분
# eig_vals	고유값 배열 → 각 고유벡터 방향으로의 분산량 (정보량)
# eig_vecs	고유벡터 행렬 → 각 열(column)이 주성분 벡터임 
eig_vals, eig_vecs = np.linalg.eig(C)
print("고유값:", eig_vals)
print("고유벡터 (주성분):\n", eig_vecs)

plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5)

origin = np.mean(X_centered, axis=0)

# 고유벡터가 2개일 경우 (2차원 데이터)
# 첫 번째 주성분(PC1), 두 번째 주성분(PC2)을 각각 그리기 위해 반복문 사용
for i in range(2):
    vec = eig_vecs[:, i]  # i번째 고유벡터(=주성분 벡터)를 열 벡터 형태로 꺼냄. i=0 → [0.71, 0.71] → PC1
    scale = eig_vals[i] ** 0.5 * 2       # 길이 조절
    plt.quiver(origin[0], origin[1], vec[0], vec[1],    # 화살표 그리기 함수
               angles='xy', scale_units='xy', scale=1, color='r')  
	# 2차원 평면에 PCA의 주성분 벡터 2개(PC1, PC2)를 빨간 화살표로 그림
	# scale = eig_vals[i] ** 0.5 * 2
	#  고유값은 해당 고유벡터 방향으로의 데이터 분산 정도
	#  분산의 제곱근은 표준편차 → 즉 실제 데이터 퍼진 거리 느낌
	#  * 2는 화살표 길이를 보기 좋게 늘리기 위한 시각화용 배율

plt.title("Principal Component Direction (Eigenvectors)")
plt.axis("equal")
plt.grid(True)
plt.show()

# PCA 선형변환 = 새 좌표계로 투영.  주성분 벡터로 투영 (변환)
X_pca = X_centered @ eig_vecs
# X_centered: 평균이 0이 되도록 중심화한 원본 데이터 행렬
# eig_vecs: 공분산 행렬의 고유벡터들
# @: 행렬 곱 연산자 (numpy에서 @는 np.dot()과 같음) 

plt.figure(figsize=(6, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title("Data projected into PCA space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.axis("equal")
plt.show()
# 이제 새로운 축(PC1, PC2) 위에 데이터를 표현함. 대부분의 정보가 첫 번째 축(PC1)에 몰려있는 걸 알 수 있다.

"""
PCA 핵심 요약
1.중심화 :  평균이 0이 되도록 데이터 이동
2.공분산 :  데이터 방향/관련성 분석
3.고윳값 분해 : 중요 방향(고유벡터) 찾기
4.선형변환 : 데이터를 새로운 축에 투영
5.차원 축소 : 정보가 많은 축만 남김
요약 : PCA는 데이터를 회전시켜서, 가장 중요한 방향으로 정렬한 후 그 방향으로만 데이터를 표현하여 정보 손실 없이 차원을 줄이는 방법이다.
"""