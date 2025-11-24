"""
VectorDB에 텍스트 파일 저장 후 유사한 텍스트 읽기

동 폴더에 있는 sample.txt 파일을 읽어서 백터화 시킨 후, 벡터 DB에 저장.
의미적 유사도 구하기
"""
import os, re, uuid     # uuid : 고유 식별 id 생성용
from typing import List # type hint 기능 제공 (가독성)
from sentence_transformers import SentenceTransformer   # 문장 단위의 의미 임베딩 라이브러리
from chromadb import PersistentClient

TXT_PATH = r".\anal11\sample.txt"
CHROMA_DIR = ".chroma_txt_demo"
COLLECTION = "docs"
MODEL_NAME = "all-MiniLM-L6-v2"     # 문장 임베딩을 위한 모델 이름

model = SentenceTransformer(MODEL_NAME)
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION)

### 기능에 따른 함수 작성하기 (유틸)
# text 파일 읽기
def read_textFunc(path:str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 존재하지 않음 : {path}")
    with open(path, 'r', encoding="utf-8", errors='ignore') as f:
        return f.read()

# 문단 단위로 분리
def split_paragraphFunc(text:str, min_len:int=20) -> List[str]:
    paras = re.split(r"\n\s*\n+", text)     # 긴 줄 기준 문단 분리
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
    return [p for p in paras if len(p) >= min_len]      # 너무 짧은 문장은 제거.

# 임베딩 백터로 처리해 저장함.
def embedFunc(texts:List[str]) -> List[List[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()   # 임베딩 벡터로 처리하면서 동시에 L2 정규화를 함.(모든 벡터의 총합이 1이 되도록.)

# text 파일을 읽어서 VectorDB에 저장
def upsert_paragraphFunc(source_path:str):
    text = read_textFunc(source_path)           # 텍스트 파일 읽기
    print(text)
    chunks = split_paragraphFunc(text)          # 읽은 텍스트를 청크(문단) 단위로 분리.
    print(chunks)
    if not chunks:
        print("저장할 문단이 없음")
        return
    ids = [str(uuid.uuid4()) for _ in chunks]   # 각 문단에 적용할 고유 id를 생성.
    print('ids : ', ids)
    embs = embedFunc(chunks)
    print(embs)
    metas = [{"source":os.path.basename(source_path), "len":len(c)} for c in chunks]  # 메타데이터 생성

    collection.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas) # VectorDB에 저장하기
    print(f"저장이 완료되었습니다. : {len(chunks)}개 문단.")

# 검색기능
def searchFunc(query:str, k:int):
    q_emb = embedFunc([query])      # 검색할 문단을 임베딩
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
        print(doc[:300] + ("..." if len(doc) > 300 else ""))          # 너무 많을 경우, 300자 까지만 보기


### 메인 매서드
if __name__ == "__main__":
    #upsert_paragraphFunc(source_path=TXT_PATH)
    print("\n검색 예 : ")
    searchFunc("스릴형 놀이기구로는 나를 즐겁게 한다", k=3)     #이 텍스트에서 벡터화된 이 문장과 거리가 제일 가까운 것을 읽음.

    
