"""
백터 데이터베이스의 데이터 삭제
"""
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction #문장을 임베딩으로 변환

embedding_fn = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
client = PersistentClient(path='.chroma')

# 컬렉션 생성
#client.delete_collection('test')
collection = client.get_or_create_collection(name='test', embedding_function=embedding_fn)

# 컬렉션에 자료 추가하기(add - insert)
collection.add(
    documents=[
        "문서 1 : 금요일 퇴근 후에 헬스장",
        "문서 2 : 잠은 언제 자나~",
    ],
    metadatas=[
        {'tag' : 'mes1'},
        {'tag' : 'mes2'}
    ],
    ids=[
        "doc1",
        "doc2"
    ]
)

# 자료 조회하기
results = collection.get(include=['documents', 'metadatas', 'embeddings'])
for doc, meta, id, emb in zip(results['documents'], results['metadatas'], results['embeddings'], results['ids']):
    print(f'id : {id}')
    print(f'document : {doc}')
    print(f'metadata : {meta}')
    print(f'embedding : {emb[:5]}')
    print(f'embedding dimension : {len(emb)}')
    print('-' * 50)

# 자료 업데이트 하기
collection.update(
    ids=['doc2'],
    documents=['문서2 : '],
    metadatas=[{'tag':'edited-mes'}],
)

# 다시 자료 조회하기
results = collection.get(include=['documents', 'metadatas', 'embeddings'])
for doc, meta, id, emb in zip(results['documents'], results['metadatas'], results['embeddings'], results['ids']):
    print(f'id : {id}')
    print(f'document : {doc}')
    print(f'metadata : {meta}')
    print(f'embedding : {emb[:5]}')
    print(f'embedding dimension : {len(emb)}')
    print('-' * 50)

# 자료 삭제하기
collection.delete(ids=['doc1'])
collection.delete(where={'tag':'edited-mes'})

# 다시 자료 조회하기
results = collection.get(include=['documents', 'metadatas', 'embeddings'])
for doc, meta, id, emb in zip(results['documents'], results['metadatas'], results['embeddings'], results['ids']):
    print(f'id : {id}')
    print(f'document : {doc}')
    print(f'metadata : {meta}')
    print(f'embedding : {emb[:5]}')
    print(f'embedding dimension : {len(emb)}')
    print('-' * 50)