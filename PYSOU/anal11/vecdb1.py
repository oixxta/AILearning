"""
- PersistentClient 
: PersistentClient(path=".chroma")는 지정한 경로에 DuckDB 기반의 저장소를 생성.
: 설치 경로 아래에 실제로 .chroma/index 같은 디렉터리가 만들어지며, 여기에 벡터, 메타, 임베딩 설정이 저장됨.
: 즉, 얘를 쓰면 메모리가 아니라 디스크 기반 DB로 영속 저장됨. (세션응 재시작해도 데이터 유지.)
: in-memory 모드도 가능 Client()로 생성하면 메모리 기반으로 동작 (휘발성)
"""

import chromadb
from chromadb import PersistentClient   # DB 접속 관리자 역할.
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import pandas as pd

print(chromadb.__file__)  # chromadb의 모든 내장 명령어 확인

# chroma는 DuckDB 기반의 벡터 저장소로 작동, 데이터를 Parquet 형식으로 저장.
client = PersistentClient(path=".chroma")   # 테이블 생성

# collection 생성 : RDBMS(관계형 데이터 베이스)의 Table과 비슷한 개념.
# collection 단위로 문서를 그룹화.
collection = client.get_or_create_collection("test")    # 기본 내장 임베딩 모델 : all-MiniLm-L6-v2
print(collection, ' ', collection.id)   # 컬렉션 구분용 아이디는 자동으로 랜덤생성.

# 문서를 벡터화해서 DB에 저장하기
texts = ['Hello world', 'Chroma is cool']
ids = ['doc1', 'doc2']
metas = [{'source' : 'greeting'}, {'source' : 'statement'}]

embedding_fn = collection._embedding_function    # chroma가 기본적으로 설정해 놓은 인베딩 함수를 반환 (텍스트 -> 벡터화 수행.)
embeddings = embedding_fn(texts)                 # texts를 벡터화 시킴.
print(type(embeddings), len(embeddings), len(embeddings[0]))    #<class 'list'> 2 384 : 타입은 리스트형, DB안 벡터 갯수는 2개, 0번째 벡처의 최대 차원 수는 384차원.
print(embeddings)

# DB 안의 내용들 확인하기
for i, vector in enumerate(embeddings):
    print(f'문서 : {texts[i]}')
    print(f'임베딩 벡터 앞 5개만 출력 : {vector[:5]}')
    print(f'차원 수 : {len(vector)}')
    print('-' * 50)
print('~' * 50)

# DB 안의 두 문장의 비교해보기 (유사성)
sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f'코사인 유사도 : {sim}')     # 유사도가 1에 근사할 수록 같은 의미.

# collection에 문서 + 벡터 + 메타데이터 저장하기
collection.add(
    documents=texts,            # 원본 데이터
    embeddings=embeddings,      # 임베딩 데이터
    metadatas=metas,            # 메타 데이터
    ids=ids,            
    #uris=['https://example.com/test1']
)
print(collection)

# collection에 저장된 자료 조회하기.
results = collection.get(include=['documents', 'metadatas'])
print(results)

for doc, meta, id in zip(results['documents'], results['metadatas'], results['ids']):
    print(f':ids : {id}')
    print(f':documents : {doc}')
    print(f':metadatas : {meta}')
    print('-' * 50)

print('저장된 문서 id 목록 : ', collection.get()['ids'])

result_vec = collection.get(include=['embeddings'])

# 첫 번째 문서의 임베딩 벡터 자료 출력하기
first_embedding = result_vec['embeddings'][0]
print('임베딩 벡터의 차원 수 : ', first_embedding[:2], ' ', len(first_embedding))
print()
for id_, embed in zip(result_vec['ids'], result_vec['embeddings']):
    print(f'id : , {id_}')
    print(f'임베딩 앞 5개 : {embed[:5]}')

# 벡터 기반 유사도 검색하기
query_text = "chroma에 대해 설명해 줘."     # 검색용 질문
query_embedding = embedding_fn([query_text])[0]    # 검색용 질문 문장을 벡터화.

# Chroma에 저장된 자료 중에서 유사 자료 검색 시작 : 
search_result = collection.query(
    query_embeddings=[query_embedding],     # 질문을 벡터로 바꾼 결과로 대입.
    n_results=2,        # 유사도가 높은 자료 2개 반환.
    include=['documents', 'metadatas', 'distances']
)
print(search_result)

# 결과 출력
for i, (doc, meta, dist) in enumerate(zip(
    search_result['documents'][0],
    search_result['metadatas'][0],
    search_result['distances'][0])):
    print(f'\n결과 : {i + 1}')
    print(f' document : {doc}')
    print(f' metatatas : {meta}')
    print(f' distances(유사도 거리) : {dist:.4f}')


# 데이터를 판다스 데이터 프레임으로 저장하기.
results = collection.get(include=['embeddings', 'metadatas', 'documents'])
df = pd.DataFrame({
    'id':results['ids'],                # collection에는 ids는 명시를 안해도 자동으로 항상 딸려옴.
    'documents':results['documents'],
    'metadatas':results['metadatas'],
    'embeddings_len':[len(e) for e in results['embeddings']]
})
print(df)
#    id       documents                metadatas  embeddings_len
#0  doc1     Hello world   {'source': 'greeting'}             384
#1  doc2  Chroma is cool  {'source': 'statement'}             384

