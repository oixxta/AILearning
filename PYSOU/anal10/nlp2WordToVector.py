"""
Word2Vec : 단어를 희소한 형태 대신, 의미를 반영한 저차원의 밀집 벡터로 표현하는 모델.

pip install gensim 필요


"""
from gensim.models import Word2Vec

sentences = [['king', 'queen', 'man', 'woman'], ['apple', 'banana', 'fruit']]   #토큰화는 미리 시켜놓음.
model = Word2Vec(sentences=sentences, vector_size=10, window=2, min_count=1, sg=1)
#                               벡터차원, 문맥 윈도우 크기, 모델 학습방법: 1(skip-gram), 0(CBOW)

print(model.wv['king'])     #단어를 백터화하고, 그 결과를 확인하기 
"""
[-0.01577653  0.00321372 -0.0414063  -0.07682689 -0.01508008  0.02469795
 -0.00888027  0.05533662 -0.02742977  0.02260065]
"""
print(model.wv.similarity('king', 'queen'))     #유사도 검사
"""
-0.042645358. 방향이 음수이면 다른 방향, 양수이면 같은 방향. 코사인 유사도를 이용해 계산됨.
"""

sentences2 = [['python', 'len', 'program', 'computer', 'say']]
model2 = Word2Vec(sentences=sentences2, vector_size=50, window=2, min_count=1, sg=1, alpha=0.0250)
# SGD(확률적 경사하강법)을 이용해 손실들 최소화 하는 방법으로 씀.
print(model2.wv)
print('인덱스 사전 : ', model2.wv.key_to_index)
print('keys : ', model2.wv.key_to_index.keys())
print('values : ', model2.wv.key_to_index.values())

vocabs = model2.wv.key_to_index    #단어사전을 기억함.
wordvec_list = [model2.wv[i] for i in vocabs]
print(wordvec_list)
print(len(wordvec_list[0]))
print(wordvec_list[0])

print(model2.wv.similarity(w1='python', w2='computer')) # 'python'과 'computer' 유사도 비교하기
print(model2.wv.most_similar('python', topn=2))     #'python'과 가장 유사한 단어 두 개만 보이기


# 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib

def plotFunc(vocabs, x, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy = (x[i], y[i]))

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xytrans = pca.fit_transform(wordvec_list)
xs = xytrans[:, 0]
ys = xytrans[:, 1]
plotFunc(vocabs, xs, ys)
plt.show()
plt.close()


# 각도 확인
import numpy as np
print(np.degrees(np.arccos(0.16563551127910614)))   # 80.46584540892889


# 유사도 순으로 정렬해서 가까움 정도를 텍스트로 표현하기(막대그래프)
target = 'python'
sim = {w:model2.wv.similarity(target, w) for w in vocabs if w != target}    #target을 제외한 나머지 것들의 similarity 계산
sort_sim = sorted(sim.items(), key=lambda x:x[1], reverse=True)
print(f"'{target}' 기준 단어별 코사인 유사성 \n")
for word, s in sort_sim:
    bar = '■' * int((s + 1) * 10)
    print(f"{word:<10}|{bar:20} ({s:.3f})")

"""
'python' 기준 단어별 코사인 유사성

program   |■■■■■■■■■■■          (0.166)
computer  |■■■■■■■■■■■          (0.125)
say       |■■■■■■■■             (-0.118)
len       |■■■■■■■              (-0.206)
"""