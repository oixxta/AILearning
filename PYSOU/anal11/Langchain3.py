"""
GPT vs Gemini key 사용 방법
"""

"""
#################### OpenAI #######################
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_template("당신은 기상 전문가야. 아래 질문에 정확히 답해 줘. <질문>{input}")
output_parser = StrOutputParser()

# LCEL chaining 기법
chain = prompt | llm | output_parser
response = chain.invoke({"input":"먹구름이 끼면 어떤 일이 벌어지니?"})
print(response)
"""

####################### Google gemini #######################
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
prompt2 = ChatPromptTemplate.from_template("당신은 기상 전문가야. 아래 질문에 정확히 답해 줘. <질문>{input}")
output_parser2 = StrOutputParser()

# LCEL chaining 기법
chain2 = prompt2 | llm2 | output_parser2
response2 = chain2.invoke({"input":"먹구름이 끼면 어떤 일이 벌어지니?"})
print(response2)
