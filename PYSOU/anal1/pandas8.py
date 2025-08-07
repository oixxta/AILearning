import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from bs4 import BeautifulSoup
import urllib.request   #연습용, 코드가 장황
import requests         #실전용, 상대적으로 코드가 간단

#웹 스크래핑
url = "https://www.kyochon.com/menu/chicken.asp"
response = requests.get(url)
response.raise_for_status()     #웹 페이지가 정상적으로 응답하는지 확인, 오류 발생 시 예외 처리를 함.

soup = BeautifulSoup(response.text, 'lxml')
print(soup)


#메뉴 이름 추출
names = [tag.text.strip() for tag in soup.select('dl.txt > dt')]
print(names)

#메뉴 가격 추출
prices = [int(tag.text.strip().replace(',','')) for tag in soup.select('p.money strong')]
print(prices)

#데이터 프레임에 넣기
myDf = DataFrame({'상품명':names, '가격':prices})
print(myDf.head(3))
print('가격 평균:', round(myDf['가격'].mean(),2))
print(f"가격 평균:{myDf['가격'].mean():.2f}")
print('가격 표준편차:', round(myDf['가격'].std(),2))
