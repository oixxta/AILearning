"""
VectorDB에 json 파일 저장 후 유사한 텍스트 읽기

동 폴더에 있는 sample.json 파일을 읽어서 백터화 시킨 후, 벡터 DB에 저장.
의미적 유사도 구하기
"""
import os, re, uuid     # uuid : 고유 식별 id 생성용
from typing import List # type hint 기능 제공 (가독성)
from sentence_transformers import SentenceTransformer   # 문장 단위의 의미 임베딩 라이브러리
from chromadb import PersistentClient
import json

JSON_PATH = r".\anal11\sample.json"
CHROMA_DIR = ".chroma_json_demo"
COLLECTION = "json_docs"
MODEL_NAME = "all-MiniLM-L6-v2"     # 문장 임베딩을 위한 모델 이름

model = SentenceTransformer(MODEL_NAME)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)

### 기능에 따른 함수 작성하기 (유틸)
# text 파일을 읽어서 VectorDB에 저장
def upsert_jsonFunc(json_path:str):
    if not os.path.exists(json_path):
        raise FileNotFoundError("파일 없음")
    with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
        if not data:
            print("자료 없음")
            return 

    ids = [item.get("id", str(uuid.uuid4())) for item in data]   # 각 문단에 적용할 고유 id를 생성.
    docs = [f"{item.get('title', '')}. {item.get('content', '')}" for item in data]
    metas = [{"title":item.get('title', ''), "source":os.path.basename(json_path)} for item in data]
    embs = model.encode(docs, normalize_embeddings=True).tolist()

    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas) # VectorDB에 저장하기
    print(f"저장이 완료되었습니다. : {len(data)}개 문단.")

# 검색기능
def searchFunc(query:str, k:int):
    q_emb = model.encode([query], normalize_embeddings=True).tolist()      # 검색할 문단을 임베딩
    res = collection.query(query_embeddings=q_emb, n_results=k)
    # 자료가 있다면, [['doc1', 'doc2']] -> [0] -> ['doc1', 'doc2']
    # 자료가 없다면, [[]] -> [0] -> [], 에러 없이 빈 리스트 반환.(예외 방지용 패턴)
    docs = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]
    ids = res.get('ids', [[]])[0]
    dists = res.get('distances', [[]])[0]

    for i, (doc, meta, _id, dist) in enumerate(zip(docs, metas, ids, dists)):
        print(f'\n[{i}] id = {_id}')
        print(f'source={meta.get("source")}, len={meta.get("len")}, distance={dist:.4f}')
        print(doc[:100] + ("..." if len(doc) > 300 else ""))          # 너무 많을 경우, 300자 까지만 보기

### 메인 매서드
if __name__ == "__main__":
    upsert_jsonFunc(json_path=JSON_PATH)
    print("\n검색 예 : ")
    searchFunc("노드와 포인터로 이루어진 자료구조 만세", k=3)     #이 텍스트에서 벡터화된 이 문장과 거리가 제일 가까운 것을 읽음.
