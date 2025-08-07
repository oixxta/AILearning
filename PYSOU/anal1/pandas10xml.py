import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

"""
# XML로 제공되는 날씨 자료 긁어와서 처리
url = "https://www.kma.go.kr/XML/weather/sfc_web_map.xml"
#data = urllib.request.urlopen(url).read()
#print(data.decode('utf8'))
soup = BeautifulSoup(urllib.request.urlopen(url), 'xml')
print(soup)


#데이터프레임으로 가공
local = soup.find_all('local')
data = []
for loc in local:
    city = loc.text
    temp = loc.get('ta')
    data.append([city, temp])

df = pd.DataFrame(data, columns=['지역', '온도'])
print(df.head(3))
df.to_csv('weather.csv', index=False)
"""

df = pd.read_csv('weather.csv')
print(df.head(2))
print(df[0:2])
print(df.tail(2))
print(df[-2:len(df)])
print()
print(df.iloc[0:2, :])
print(df.loc[1:3,['온도']])
print(df.info())
print(df['온도'].mean())
print(df.sort_values(['온도'], ascending=True))