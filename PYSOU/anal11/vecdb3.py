import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

"""
벡터 DB의 로컬 파일 저장, 유사도 기반 문장 분류, 문장 클러스터링
"""
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

client = chromadb.PersistentClient(path='.chroma_db')
collection = client.get_or_create_collection(name='simple_texts', metadata={'hnsw:space': 'cosine'})

texts = [
    '사과는 과일이다',
    '파이썬은 프로그래밍 언어다',
    '해는 동쪽에서 뜬다',
    '나는 망고를 좋아한다'
]

ids = [str(i) for i in range(len(texts))]
print(ids)

# 임베딩 모델을 로딩하기
model = SentenceTransformer('all-MiniLM-L6-v2')
print('현재 모델 임베딩 차원 수 : ', model.get_sentence_embedding_dimension())
print(model)

embeddings = model.encode(texts)
print(embeddings[:3])
print(model.encode(texts).shape)

# 벡터DB에 저장
collection.add(documents=texts, ids=ids, embeddings=embeddings)

# 저장된 벡터 출력
record = collection.get(ids=['0'], include=['embeddings', 'documents'])
print('첫 번째 문서 : ', record['documents'][0])
print('첫 번째 문서의 임베딩 벡터 값 : ', record['documents'][0][:10])

# 유사 문장 검색
query = '파이썬은 무엇인가?'
query_vector = model.encode([query]).tolist()
print(model.encode(query).shape, model.encode([query]).shape)

result = collection.query(
    query_embeddings=query_vector,
    n_results=2,
    include=['documents', 'distances']  #검색 결과에 포함시킬 항목을 지정.
)
print('검색 결과 : ')
for doc, dist in zip(result['documents'][0], result['distances'][0]):
    print(f'- 문장 : {doc}(유사도 거리 : {dist:.4f})')

# 유사도 기반 문장 분류기 작성
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

texts2 = [
    '나는 과일을 즐겨 먹는다.', '바나나는 내가 좋아하는 과일이다.', '파이썬은 훌륭한 프로그래밍 언어야.',
    '나는 가끔 파이썬을 이용해 코딩을 하고 있어.', '사과와 바나나는 둘 다 맛이 좋아',
    '파이썬으로 코딩히면 이해가 잘 돼.', '나는 망고 스무디를 즐겨 마신다.', '과일은 건강한 간식이다.',
    '나는 열대 과일을 특히 좋아한다.'
]

mymodel = SentenceTransformer('all-MiniLM-L6-v2')
myembeddings = model.encode(texts2)
print(myembeddings[:3])

# 클러스터링 실시
n_clusters = 3  # k=3
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans_model.fit_predict(myembeddings)
print(labels)

for idx, (text, label) in enumerate(zip(texts2, labels)):
    print(f'[군집 {label}] ({text})')

# 군집 결과 시각화
# 384차원을 2차원으로 축소 (주성분분석, PCA를 활용해서.)
pca = PCA(n_components=2)
reduced = pca.fit_transform(myembeddings)
print(reduced)

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i in range(n_clusters):
    cluster_points = reduced[labels == i]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], color=colors[i % len(colors)], label = f"Cluster{i}")
plt.title("문장 군집화(PCA) 시각화")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.grid(True)
plt.show()
plt.close()

# 클러스터별 대표 문장 추출
# 두 벡터 집합 간의 가장 가까운 쌍(인덱스와 거리)을 구할 때 사용하는 함수
print('클러스터별 대표 문장 : 중심에 가장 가까움')
for i  in range(n_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_embeddings = myembeddings[cluster_indices]  # i번째 클러스터에 속한 문장들의 벡터 모음

    center = kmeans_model.cluster_centers_[i].reshape(1, -1)
    closets_idx, _ = pairwise_distances_argmin_min(center, cluster_embeddings)
    closets_text = texts2[cluster_indices[closets_idx[0]]]  # 클러스터 중심에 가장 가까운 문장을 texts2 리스트에서 찾음.

    print(f'[Cluster{i}] {closets_text}')

# 클러스터별 전체 문장
for i in range(n_clusters):
    print(f'\n[Cluster{i}]')
    for idx, (text, label) in enumerate(zip(texts2, labels)):
        if label == i:
            print(f' - {text}')
