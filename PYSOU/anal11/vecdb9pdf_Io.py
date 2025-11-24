"""
VectorDB에 PDF 파일 저장 후 유사한 텍스트 읽기

동 폴더에 있는 sample.pdf 파일을 읽어서 백터화 시킨 후, 벡터 DB에 저장.
의미적 유사도 구하기
"""
import os, re, uuid     # uuid : 고유 식별 id 생성용
from typing import List # type hint 기능 제공 (가독성)
from sentence_transformers import SentenceTransformer   # 문장 단위의 의미 임베딩 라이브러리
from chromadb import PersistentClient
import pypdf

PDF_PATH = r".\anal11\sample.pdf"
CHROMA_DIR = ".chroma_txt_demo"
COLLECTION = "docs"
MODEL_NAME = "all-MiniLM-L6-v2"

### 기능에 따른 함수 작성하기 (유틸)
# pdf 파일 읽기
def read_pdfFunc(path:str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 존재하지 않음 : {path}")
    text_pages = [] # 페이지별 텍스트 저장 리스트
    try:
        with open(path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                text_pages.append(txt)
        return "\n\n".join(text_pages)
    except Exception:
        raise RuntimeError("PDF 추출 실패")
    
# 문단 단위로 분리
def split_paragraphFunc(text:str, min_len:int=40) -> List[str]:
    chunks = re.split(r"\n\s*\n+", text)     # 긴 줄 기준 문단 분리
    chunks = [re.sub(r"\s+", " ", p).strip() for p in chunks]
    return [p for p in chunks if len(p) >= min_len]      # 너무 짧은 문장은 제거.

# 임베딩 모델 로딩
def embedderFunc(name:str=MODEL_NAME):
    return SentenceTransformer(name)

# 임베딩 백터로 처리해 저장함.
def embedFunc(model, texts:List[str]) -> List[List[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()   # 임베딩 벡터로 처리하면서 동시에 L2 정규화를 함.(모든 벡터의 총합이 1이 되도록.)

def get_collectionFunc(chroma_dir:str, name:str):
    client = PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(name)

# pdf 파일을 읽어서 VectorDB에 저장
def upsert_pdfFunc(source_path:str):
    full_text = read_pdfFunc(PDF_PATH)           # 텍스트 파일 읽기
    print(full_text)

    if not full_text.strip():
        print('pdf에서 추출된 자료가 없음')
        return 0
    
    chunks = split_paragraphFunc(full_text, min_len=40)          # 읽은 텍스트를 청크(문단) 단위로 분리.
    print(chunks)
    if not chunks:
        print("저장할 문단이 없음")
        return 0
    
    model = embedderFunc(MODEL_NAME)
    embs = embedFunc(model, chunks)
    print(embs)

    metas = []

    for c in chunks:
        metas.append(
            {
                "source":os.path.basename(source_path),
                "len" : len(c)
            }
        )

    collection = get_collectionFunc(CHROMA_DIR, COLLECTION)
    ids = [str(uuid.uuid4()) for _ in chunks]   # 각 문단에 적용할 고유 id를 생성.
    print('ids : ', ids)
    embs = embedFunc(chunks)
    print(embs)
    collection.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas) # VectorDB에 저장하기
    print(f"저장이 완료되었습니다. : {len(chunks)}개 문단.")
    return len(chunks)

# 검색기능
def searchFunc(query:str, k:int):
    model = embedderFunc(MODEL_NAME)
    q_emb = embedFunc(model, [query])      # 검색할 문단을 임베딩
    collection = get_collectionFunc(CHROMA_DIR, COLLECTION)
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
    n = upsert_pdfFunc(PDF_PATH)
    print("\n저장된 문단 수 : {n}")
    if n:
        searchFunc("강남에 지역 문화적 뉘앙스를 더하면 로컬 개발자들의 개성이 살아날 수 있다", k=3)     #이 텍스트에서 벡터화된 이 문장과 거리가 제일 가까운 것을 읽음.
