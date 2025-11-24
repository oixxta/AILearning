from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# 랭체인 기본 : LLM 연결하기 (추상화)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
response = llm.invoke("what is love?")
print(response)


# PromptTemplate 사용 : 기본 프롬프트 + 변수 삽입
mytemplate = """너는 한국어 전문가야. 아래 내용으로 5행 이내의 아름다운 시ㅏ를 작성해 줘. "{content}"
"""

print("template : ", mytemplate)

prompt = PromptTemplate(
    input_variables=["content"],
    template=mytemplate
)

fill_prompt = prompt.format(content="가을하늘")
print("fill_prompt : ", fill_prompt)

porm_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
poem_response = porm_llm.invoke(fill_prompt)
print(poem_response.content)



# tool 사용 (tool + Agent 구조)
print("계산기 툴 작성")
from langchain.tools import tool
from langchain.agents import create_agent

# tool 정의하기
@tool
def myCulc(expression:str) -> str:
    # tool에는 """" 간단한 설명 """ 을 반드시 넣어야 함.
    """간단한 사칙연산 수식을 계산하고 '수식=값' 형태의 문자열로 반환한다"""
    try:
        result = eval(expression)   # eval("6 + 7")
        return f"{expression} = {result}"
    except Exception as e :
        return f"fail to calculate : {e}"
    
tools = [myCulc]
model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

agent = create_agent(
    model = model,
    tools = tools,
    system_prompt=(
        "너는 수학 계산을 도와주는 어시스턴트야."
        "가능하면 myCulc 툴을 사용해 정확한 값을 계산해 줘"
        "출력은 형식을 지켜서 답해. 수식 = 값"
    ),
)

question = "7 * (3 + 2) / 2 는 얼마야?"
result = agent.invoke({
    "messages":[
        {"role" : "user", "content" : question}
    ]
})
#print(result)
last_msg = result["messages"][-1]
print("최종 답변 : ", last_msg.content)

