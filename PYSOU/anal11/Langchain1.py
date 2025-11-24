"""
*** LangChain 핵심 기능 로드맵 ***
 1. LLM (대형 언어 모델)
 2. PromptTemplate (프롬프트 설계)
 3. DocumentLoader (문서 읽기)
 4. TextSplitter (문서 나누기)
 5. Embeddings (문장을 벡터로)
 6. VectorStore (벡터 저장소, 예: Chroma)
 7. Retriever (연관 문서 검색)
 8. Chain (작업 흐름 연결)
 9. Agent & Tools (툴 조합)
 10. Memory (대화 상태 유지)

Langchain은 언어 모델(LLM)을 활용해 다양한 어플리케이션을 개발할 수 있는 프레임워크
"""
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


### 프롬프트 설계
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI KEY가 없음.")

### 문서 로딩
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
docs = TextLoader('sample.txt', encoding='utf-8').load()
print(f'문서 갯수 : {len(docs)}')
print(docs)
print(docs[0].page_content)

### 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(documents=docs)
print(f'생성된 청크 수 : {len(chunks)}')
print(chunks)
print(chunks[0].page_content)

### 임베딩 & 벡터 스토어
def format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)

from langchain_core.runnables import RunnableLambda
docs_runnable = RunnableLambda(lambda _: chunks)  # 고정 청크를 Runnable로 래핑

prompt = ChatPromptTemplate.from_template(
    '답하세요\n'
    '{context}\n'
    '질문:{question}'
)

# 고정된 데이터는 RunnableLambda로 감싸야 한다.
chain =(
    {   # 1) 입력 데이터 구성
        # context 생성 규칙
        # docs_runnable: 실행될 때 항상 'chunks' 리스트를 반환하는 RunnableLambda
        # docs_runnable | RunnableLambda(format_docs)
        #      : chunks 리스트를 문서 문자열(context)로 변환
        'context':docs_runnable | RunnableLambda(format_docs),
        # question 생성 규칙 : 사용자 입력(question)을 그대로 다음 단계로 전달
        'question':RunnablePassthrough()
    }
    # 2) PromptTemplate 적용 : {context, question}을 prompt 템플릿에 채워 넣음
    | prompt
    # 3) 템플릿이 완성되면 llm을 호출해 답변 생성
    | llm
    # 4) LLM 결과에서 message 객체 대신 문자열만 추출하여 텍스트로 출력
    | StrOutputParser()
)

user_question = "현대자동차에 대해 설명해"
answer = chain.invoke(user_question)
print(f'질문 : {user_question}')
print(f'대답 : {answer}')
