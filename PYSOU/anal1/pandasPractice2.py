import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from bs4 import BeautifulSoup
import requests         #실전용, 상대적으로 코드가 간단
import matplotlib.pyplot as plt
import seaborn as sns
import time

#KFC 메뉴 페이지 자료 읽기
#출력 : 메뉴명, 가격, 설명
#통계 : 건수, 가격평균, 표준편차, 최고가격, 최저가격, 기타,
#시각화도 하기

#웹 스크래핑
url = "https://bongousse.com/Menu_list.asp"
headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
response = requests.get(url, headers=headers)
response.raise_for_status()
menuData = response.json()

myDf = pd.DataFrame(menuData)
print(myDf)

#메뉴 이름 추출
