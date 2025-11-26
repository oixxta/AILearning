"""
멀티 모달 기본 이해 - 이미지를 읽어서(이미지 캡션) LLM이 설명하게 하기.

사용할 이미지 : person.jpeg

"""
import base64   # 이미지 파일을 base64 배열로 전환해야 하기 때문에 필요
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# 이미지를 base64 Data URL(data:image/...)로 변환하는 함수
def encode_image_to_data_urlFunc(img_path:str) -> str:
    # Gemini는 jpg는 지원하지 않음. 따라서, jpg를 jpeg로 바꾸게 함.
    ext = Path(img_path).suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")    # base64로 인코딩 한 뒤 문자열로 반환
    
    return f"data:image/{ext};base64,{b64}" # Langchain이 요구하는 형식으로 반환


# 이미지를 llm이 설명하도록 하는 함수
def desc_imageFunc(img_path:str) -> str:
    img_url = encode_image_to_data_urlFunc(img_path)    # 이미지를 data64로 변환
    print(img_url)

    # HumanMessage 타입 형식으로 메시지 생성
    msg = HumanMessage(
        content=[
            # LLM이 수행할 요청
            {"type" : "text", "text" : "현재 이미지에 보이는 내용을 다섯 문장 이내로 설명해줘."},

            # 이미지 제공
            {"type" : "image_url", "image_url" : {"url":img_url}},
        ]
    )

    result = vision_llm.invoke([msg])
    return result.content       # 순수 텍스트만.


if __name__ == "__main__":
    img_path = "person.jpeg"

    print("사진 설명 : ")
    print(desc_imageFunc(img_path))