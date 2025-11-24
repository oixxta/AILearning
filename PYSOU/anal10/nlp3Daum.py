"""
뉴스 자료를 읽어서 형태소를 분석 후, 단어 유사도 확인하기.

"""

import pandas as pd

from konlpy.tag import Okt
okt = Okt()

with open('daumnews.txt', 'r', encoding='utf8') as f:
    lines = f.read().splitlines()    # split('\n') 대신 splitlines() 사용 권장

# print(lines)  

word_freq = { }   # 명사만 추출해 단어 수 확인. {'공동':3, ...}

for line in lines:
    nouns = [word for word, tag in okt.pos(line) if tag == 'Noun' and len(word) > 1]
    for noun in nouns:
        word_freq[noun] = word_freq.get(noun, 0) + 1  # dict.get()으로 간결하게

print((word_freq))    # {'세계': 1, '주요': 3, '완성': 2, '기업': 2, 

# 단어 건수별 내림차순 정렬한 후 DataFrame에 저장
# sortData = sorted(word_freq.items(), key=lambda dul:dul[1], reverse=True)
  # key=lambda a: a[1]	각 튜플에서 두 번째 요소 (a[1], 즉 빈도수)를 기준으로 정렬

# 참고 : 두 가지 기준으로 정렬 : 첫 번째: 빈도수 → 내림, 두 번째: 단어 → 오름
sortData = sorted(
    word_freq.items(),
    key=lambda dul: (-dul[1], dul[0])
)

print(sortData)

df = pd.DataFrame(sortData, columns=['단어', '빈도수'])
print(df.head(10), len(df), df.shape)      # 상위 10개만 보기  196 (196, 2)
        #     단어  빈도수
        # 0   스텔    8

# csv 파일로 저장
df.to_csv('word_freq.csv', index=False, encoding='utf-8-sig')

print('\ncsv 파일 읽기');
df = pd.read_csv('word_freq.csv', encoding='utf-8-sig')
print(df.head(3))

# print('\nword2vec을 사용해 단어 간 유사도 확인하기 ------------')
# LineSentence()는 텍스트 파일을 한 줄씩 읽어서 단어 리스트로 변환하는 클래스.
# 그래서 csv 파일 말고 txt 파일이 필요하다.

# 명사만 추출하여 문장 파일로 저장
with open('word_freq.txt', 'w', encoding='utf-8') as f:
    for line in lines:
        # 형태소 분석 시 stem=True로 동사는 원형으로 추출
        tokens = okt.pos(line, stem=True)
        # 명사 또는 동사(길이 2 이상)만 선택
        words = [word for word, tag in tokens if tag in ['Noun', 'Verb'] and len(word) > 1]
        if words:
            f.write(' '.join(words) + '\n')
# 미국 주식 투자자 미국 주식 느끼다 <== 이런 형태로 txt 파일이 생성됨

from gensim.models import word2vec

sentences = word2vec.LineSentence('word_freq.txt')
print(sentences)

model = word2vec.Word2Vec(sentences=sentences, vector_size=100, window=10, min_count=1, sg=1)
  # 100 차원의 벡터를 만들고 주ㅡ변 단어는 앞뒤로 10개 까지 참조. 단어 빈도수는 1개 이상이면 참여 
  # sg=1 : Skip-Gram 알고리즘. 작은 데이터셋에서 더 정확한 결과 도출
print(model)  # Word2Vec<vocab=172, vector_size=100, alpha=0.025>

# 학습된 모델 저장
model.save("wordmodel.model")
# 모델 로드 : 확장자는 .model, .bin, .txt 모두 가능하지만 .model은 Gensim 전용.
model = word2vec.Word2Vec.load("wordmodel.model")

print(model.wv.index_to_key[:5])    # 학습된 단어들 중 상위 5개 단어 출력

print('부동산' in model.wv.key_to_index)   # 먼저 단어 목록을 확인 True면 있음, False면 없음
print(model.wv.most_similar('주식시장'))     #  '부동산'과 유사한 단어 출력

# 두 단어의 벡터를 더한 결과에 가장 가까운 단어를 출력. 즉, "부동산"이라는 의미와 가장 유사한 단어 찾음
print(model.wv.most_similar(positive=['주식시장','한국'], topn=3))   # [('한국', 0.562566), ...]

# 부동산과 의미적으로 반대 방향에 있는 단어들
print(model.wv.most_similar(negative=['주식시장'], topn=3))  # [('결정', 0.014099, ('장사', -0.014231), ...
# 수학적 유사도(Cosine Similarity)를 기준으로 하기 때문에, 결과 해석은 항상 문맥에 따라 판단해야 한다

# "king" - "man" + "woman" = "queen" 유형 만들기
print(model.wv.most_similar(positive=['주식시장', '한국'], negative=['한국']))
  # '미국 부동산'에서 '미국'을 빼고 '한국'을 넣으면, 한국 부동산과 의미가 비슷한 단어는? 이라는 뜻이 됨.
print(model.wv.most_similar(positive=['주식시장', '한국'], negative=['한국']))
  # 미국 소득세 대신 한국 소득세와 관련된 단어를 유추

# 참고 : Word2Vec (Skip-Gram 또는 CBOW)은 기본적으로 신경망 기반 모델이다.
# 이 신경망은 학습 과정에서 경사하강법을 사용하여 단어 벡터(embedding)를 학습한다.
# * 내부 구조 요약 (Skip-Gram 기준)
#   - 입력: 중심 단어 (예: 부동산)
#   - 출력: 주변 단어들 예측 (예: 지급, 소득세, 기준일 ...)
#   - 구조: 입력층 → 은닉층(단어 벡터) → 출력층(softmax 또는 네거티브 샘플링)
#   - 손실(loss) 계산: 실제 주변 단어와 예측 값의 차이
#   - 경사하강법으로 손실을 줄이도록 단어 벡터를 조금씩 조정
# epochs=10(학습 횟수), seed=42, alpha=0.03, min_alpha=0.0005 : 초기 학습률 0.03 → 점차 0.0005까지 감소

# 유사도 기반 단어 관계 시각화 (PCA)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import platform

# 시스템별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows: 맑은 고딕
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'      # Mac: 애플 고딕
else:
    plt.rcParams['font.family'] = 'NanumGothic'     # Linux: 나눔고딕 등

target_word = '부동산'     # 기준 단어

# 유사 단어 Top 10 가져오기 (단어 + 유사도 점수)
similar_words = model.wv.most_similar(target_word, topn=10)
print('similar_words : ', similar_words[0])  # ('한국', 0.560269355773) 단어와 유사도점수

# 단어 리스트 만들기 (기준 단어 + 유사 단어들)
# word, _ ← 각 튜플을 두 부분으로 나눔. 
# word: 단어 문자열 (예: '부동산'), _ : 유사도 점수 (0.91 등) → 사용하지 않으므로 _로 무시
words = [target_word] + [word for word, _ in similar_words]

# 단어 벡터 추출
word_vectors = [model.wv[word] for word in words]
print('word_vectors : ', word_vectors[0])  # [-0.00612738  0.0039777 ...

# PCA로 2차원 축소
pca = PCA(n_components=2)
points = pca.fit_transform(word_vectors)
print('points : ', points[0])  # [-0.0038808  -0.00331953]

# 시각화
plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    x, y = points[i]
    plt.scatter(x, y, color='blue' if i == 0 else 'black')
    plt.text(x + 0.01, y + 0.01, word, fontsize=12,
             color='red' if i == 0 else 'black')  # 기준 단어는 빨간색

plt.title(f"Word2Vec 유사 단어 시각화 (기준 단어: '{target_word}')")
plt.grid(True)
plt.show()


from sklearn.cluster import KMeans
import numpy as np

# 모델에 존재하는 단어만 필터링
filtered_words = [word for word in words if word in model.wv.key_to_index]
vectors = [model.wv[word] for word in filtered_words]

# KMeans 클러스터링
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(vectors)

# PCA 축소
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)
centers = pca.transform(kmeans.cluster_centers_)   # 클러스터 중심점을 PCA 축소

# 색상 목록
colors = ['red', 'blue', 'green', 'orange', 'purple']

# 시각화
plt.figure(figsize=(10, 7))

for i, word in enumerate(filtered_words):
    x, y = reduced_vectors[i]
    plt.scatter(x, y, color=colors[labels[i]], s=120, edgecolor='black')
    plt.text(x + 0.005, y + 0.005, word, fontsize=12)

# 클러스터 중심점 그리기
for i, (cx, cy) in enumerate(centers):
    plt.scatter(cx, cy, color=colors[i], s=200, marker='X', edgecolor='black', label=f'Cluster {i+1}')

plt.title("Word2Vec 단어 의미 군집화 (KMeans + PCA + 중심점 표시)")
plt.legend(title='군집')
plt.grid(True)
plt.tight_layout()
plt.show()


# 추가 (군집별 단어 리스트 출력)
from collections import defaultdict

cluster_dict = defaultdict(list)
for word, label in zip(filtered_words, labels):
    cluster_dict[label].append(word)

for cid, word_list in cluster_dict.items():
    print(f"Cluster {cid+1}: {', '.join(word_list)}")
        # Cluster 3: 배당, 소득세
        # Cluster 1: 한국, 다른
        # Cluster 2: 하다, 되다, 장기, 넘다, 권리, 걷다, 배당금



# 덴드로그램 그리기  ----------------------
# 단어들을 의미적으로 가까운 순서대로 묶어 나가는 계층적 군집 분석 결과
# 수직축: 단어들 사이의 거리(유사하지 않음) → 값이 낮을수록 가까움, 높을수록 멀리 떨어짐
# 수평축: 단어 목록 → 실제 군집 대상 단어들
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 단어가 모델에 있는지 확인
filtered_words = [word for word in words if word in model.wv.key_to_index]
# 벡터 추출
vectors = np.array([model.wv[word] for word in filtered_words])

# 계층적 클러스터링 (거리 계산 방식: 'ward', 'average', 'complete' 등)
linkage_matrix = linkage(vectors, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=filtered_words, leaf_font_size=12)
plt.title("Word2Vec 단어 계층 군집 (Dendrogram)")
plt.xlabel("단어")
plt.ylabel("거리")
plt.tight_layout()
plt.show()