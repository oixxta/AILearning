"""
벡터 DB : 크로마 DB의 백터 폴더 생성 예시

크로마 DB의 저장 구조
collection : 문서의 논리적 그룹
document : 저장하고 싶은 텍스트
embedding : 문서를 숫자 벡터로 변환한 값
metadata : 부가적인 정보 (출처, 날짜 등)
id : 고유 식별자
저장 방식 : 인메모리 or DuckDB + Parquet 파일 기반 저장

.chroma 폴더 생성, chroma.sqlite3 파일 생성.

"""
import chromadb
from chromadb import PersistentClient   # DB 접속 관리자 역할.

client = PersistentClient(path=".chroma")
collection = client.get_or_create_collection("test")

