# 숫자 이미지 데이터에 K-평균 알고리즘 사용하기

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()      # 64개의 특징(feature)을 가진 1797개의 표본으로 구성된 숫자 데이터
print(digits.data.shape)  # (1797, 64) 64개의 특징은 8*8 이미지의 픽셀당 밝기를 나타냄

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)  # (10, 64)  # 64차원의 군집 10개를 얻음

# 군집중심이 어떻게 보이는지 시각화

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest')
plt.show()  # 결과를 통해 KMeans가 레이블 없이도 1과 8을 제외하면 
# 인식 가능한 숫자를 중심으로 갖는 군집을 구할 수 있다는 사실을 알 수 있다. 

 

# k평균은 군집의 정체에 대해 모르기 때문에 0-9까지 레이블은 바뀔 수 있다.
# 이 문제는 각 학습된 군집 레이블을 그 군집 내에서 발견된 실제 레이블과 매칭해 보면 해결할 수 있다.
from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# 정확도 확인
from sklearn.metrics import accuracy_score
print(accuracy_score(digits.target, labels))  # 0.79354479


# 오차행렬로 시각화
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()  # 오차의 주요 지점은 1과 8에 있다.

# 참고로 t분포 확률 알고리즘을 사용하면 분류 정확도가 높아진다.

from sklearn.manifold import TSNE

# 시간이 약간 걸림
tsne = TSNE(n_components=2, init='random', random_state=0)
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy

print(accuracy_score(digits.target, labels))  # 0.93266555

 
"""
참고 : 위 소스에 대한 이해를 설명 추가 -------------------

tsne = TSNE(n_components=2, init='random', random_state=0)
  TSNE = t-Distributed Stochastic Neighbor Embedding : 고차원 데이터를 2차원으로 줄여서, 비슷한 데이터는 가깝게, 다른 데이터는 멀게 만든다.
  n_components=2 : 2차원으로 축소하겠다는 뜻 (그래서 시각화가 가능)
  init='random' : 초기 배치를 랜덤으로 한다는 의미.
  random_state=0 : 랜덤성을 고정해서, 매번 똑같은 결과를 얻을 수 있게 한다.

digits_proj = tsne.fit_transform(digits.data) 
   TSNE로 digits 데이터셋을 변환한다.
  digits.data 는 8×8 이미지(= 64차원)를 1줄로 펴서 저장함.  그걸 TSNE에 넣으면 원래 64차원이었던 걸 2차원으로 줄여준다!
  digits_proj 는 결과다. shape은 (샘플 수, 2) 예를 들어, (1797, 2) 이런 형태가 된다. 즉, 각 숫자 이미지가 (x, y) 두 좌표로 표현된 것.

kmeans = KMeans(n_clusters=10, random_state=0)
  KMeans 객체를 만든다.
  KMeans = 대표적인 군집 분석(클러스터링) 알고리즘
  n_clusters=10 → 군집 개수를 10개로 정했다. (0~9 숫자 이미지라서 당연히 10개 군집을 기대.)
  
4. clusters = kmeans.fit_predict(digits_proj)
  KMeans로 군집 분석을 실행하고, 결과를 예측한다. fit_predict() 함수는 fit: 데이터를 군집화하고 (중심점 찾기) predict: 각 데이터가 어느 군집에 속하는지 레이블을 출력.
  clusters : shape은 (샘플 수,) 각 데이터가 몇 번째 클러스터에 속하는지 0~9 사이 숫자로 나타난다.

간단히 말해 고차원 데이터를 TSNE로 2차원으로 줄이고, 줄인 결과를 KMeans로 10개의 군집으로 나눈다!
64차원 (digit 이미지)  
   ↓  (TSNE: 2D로 축소)
2차원 평면 위 점들  
   ↓  (KMeans: 10개 그룹으로 나누기)
군집 결과 (0~9)
"""