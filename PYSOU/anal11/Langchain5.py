# 랭체인 에이전트 툴 구성 : langchain agent + tool + LLM
# LLM에게 상황을 분석하게 하고 -> 필요한 툴을 LLM이 스스로 선택한 후 툴을 실행하고 -> 결과를 반영해
# 다시 답을 만드는 '멀티 턴 자동의사 결정' 로직 구성. ReAct
# ReAct 프롬프트는 AI가 스스로 판단하고, 도구를 선택하고, 연속된 reasoning을 할 수 있게 만드는 설계도다.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 0) 공통 유틸 --LLM이 생성한 응답 문자열을 정리해 주는 헬퍼 함수
# Gemini 2.x/2.5/Flash 계열의 특징 때문. 같은 답변을 2번 반복하는 버그/패턴이 자주 발생함
def clean_answer(t: str) -> str:
    # 1) 입력 객체를 무조건 문자열로 만들고 앞뒤 공백 제거
    t = str(t).strip()    # (ChatMessage 객체일 수도 있어서 str()으로 통일)

    # 2) 전체 문자열이 정확히 두 번 반복된 패턴인지 검사
    #    예: "ABC\nABC\n" 같은 경우 → 절반으로 잘라서 중복 제거
    if len(t) % 2 == 0 and t[:len(t)//2] == t[len(t)//2:]:
        t = t[:len(t)//2].strip()   # 앞 절반만 사용

    # 3) 문단 단위로 중복 제거 : "\n\n" 기준으로 문단 나눠서 이미 본 문단은 제외
    out, seen = [], set()
    for p in t.split("\n\n"):
        p = p.strip()
        if p and p not in seen:     # 비어있지 않고 중복되지 않은 문단만 추가
            seen.add(p)
            out.append(p)

    # 4) 최종 문단들을 공백 한 칸으로 연결
    merged = " ".join(out)
    cleaned = merged.replace("**", "").replace("*", "")   # 마크다운 기호 제거 (**, *)
    return cleaned.strip()

# 프롬프트를 받아 LLM을 호출하고, 결과를 clean_answer로 정리하는 공통 함수
def run_llm(prompt: str) -> str:
    resp = llm.invoke(prompt)
    # ChatMessage면 content, 아니면 그대로 문자열을 반환
    return clean_answer(getattr(resp, "content", resp))


# 1) Tools ------------
@tool     # 이 함수를 LangChain Tool로 등록
def math_helper(question: str) -> str:
    """수학 문제를 단계별로 풀이하고 마지막 줄에 정답을 한 번 더 적어 주는 도우미."""
    prompt = (     # 수학 문제 전용 프롬프트 구성
        "너는 수학 풀이를 잘하는 모범생이야.\n"
        "아래 수학 문제를 단계별로 풀고, 마지막 줄에 정답을 적어 줘.\n\n"
        f"문제: {question}\n"  # 사용자의 질문(수학 문제)을 그대로 끼워 넣기
        "풀이:"
    )
    return run_llm(prompt)    # 공통 run_llm 함수로 LLM 호출 + 정리된 답변 반환

@tool
def code_helper(question: str) -> str:
    """코딩/프로그래밍 질문에 대해 1)설명 2)예제 코드 3)중요한 포인트 순서로 답하는 도우미."""
    prompt = (
        "너는 전문 프로그래머야.\n"
        "아래 요청에 대해 1)간단한 설명 2)예제 코드 3)중요한 포인트 순서로 답변해 줘.\n\n"
        f"요청: {question}\n"
        "답변:"
    )
    return run_llm(prompt)

@tool
def general_helper(question: str) -> str:
    """일반 개념/이론을 이해하기 쉽게 3~4문장으로 설명하는 도우미."""
    prompt = (
        "너는 친절한 AI 도우미야.\n"
        "아래 질문에 대해 3~4문장으로 설명해 줘.\n"
        f"질문: {question}\n"
        "답변:"
    )
    return run_llm(prompt)

tools = [math_helper, code_helper, general_helper]  # Agent에게 넘겨줄 Tool 목록


# 2) Agent
prompt = ChatPromptTemplate.from_messages(   # Agent용 전체 프롬프트 템플릿 정의
  [
    (
      "system",    # 시스템 메시지: Agent의 역할과 Tool 사용 규칙 정의
      "너는 사용자의 질문을 보고 적절한 툴을 선택하는 에이전트다.\n"
      "- 수식, 계산, 더하기/빼기/곱하기/나누기, '+', '-', '*', '/' 등이 보이면 math_helper를 사용해.\n"
      "- '함수', '클래스', Python/Java/C언어/JavaScript 등의 단어가 보이면 code_helper를 사용해.\n"
      "- 그 외의 일반적인 개념/이론/설명은 general_helper를 사용해.\n"
      "하지만 툴에서 반환된 텍스트를 그대로 복사해서 출력하지 말고, "
      "툴의 내용을 참고해 최종 답변을 한국어로 깔끔하게 작성해.\n"
    ),
    # 이전 대화 내용을 주입하기 위한 placeholder(자리표시자)
    # (이전까지의 대화 기록(사람 → AI → 사람 → AI...))
    # 이전 대화 히스토리를 넣기 위한 자리. 대화의 연속성과 문맥 유지
    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{input}"),     # 현재 사용자가 입력한 질문이 들어가는 자리
    # ask("3 더하기 5는?") 라고 하면 ("human", "{input}")에 ("human", "3 더하기 5는?")가 됨

    # Agent가 Tool호출등 중간 사고과정을 쌓는 내부 메모. Agent가 스스로 결정 내리기 위한 메모장역할
    MessagesPlaceholder(variable_name="agent_scratchpad"),
  ]
)

# Tool-calling Agent 생성 (LLM + Tools + Prompt 연결)
agent = create_tool_calling_agent(
    llm=llm,  tools=tools, prompt=prompt,
)

agent_executor = AgentExecutor(   # Agent를 실제로 실행하는 래퍼
    agent=agent,      # 방금 만든 Agent
    tools=tools,      # 동일한 Tool 목록 전달
    verbose=False,    # True면 내부 Tool 호출 로그를 콘솔에 출력, False면 숨김
)

# 3) 간단 테스트 래퍼 --------------------
chat_history = []      # 대화 히스토리(간단 버전) 저장용 리스트

def askFunc(q: str):   # 한 번의 질의를 실행하는 헬퍼 함수
    print("\n==============================")
    print("질문:", q)                   # 콘솔에 질문 출력
    result = agent_executor.invoke(    # AgentExecutor를 통해 Agent 호출
        {
            "input": q,                 # 프롬프트 템플릿의 {input} 에 매핑될 값
            "chat_history": chat_history,  # 이전 턴들의 대화 히스토리
        }
    )
    print("\n[최종 답변]")
    print(result["output"])               # Agent가 최종적으로 생성한 답변 출력
    chat_history.append(("human", q))           # 히스토리에 사용자 질문 추가
    chat_history.append(("ai", result["output"]))  # 히스토리에 AI 답변 추가


q1 = "3 더하기 5 곱하기 2는 얼마인가?"
askFunc(q1)

q2 = "자바로 숫자들의 평균을 구하는 코드를 보여줘."
askFunc(q2)

q3 = "가을과 겨울의 차이는 무엇인가?"
askFunc(q3)

