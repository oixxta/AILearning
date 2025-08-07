import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from bs4 import BeautifulSoup
import urllib.request   #연습용, 코드가 장황
import requests         #실전용, 상대적으로 코드가 간단


#XML 문서 처리
with open(r'C:\Users\acorn\Desktop\work\myNumpyPractice\PYSOU\my.xml', mode='r', encoding='utf-8') as f:
    xmlfile = f.read()
    print(xmlfile)

soup = BeautifulSoup(xmlfile, 'lxml')
print(soup.prettify())

itemTag = soup.find_all('item')
print(itemTag)
print(itemTag[0])
print()
nameTag = soup.find_all('name')
print(nameTag[0]['id'])

for i in itemTag:
    nameTag = i.find_all('name')
    for j in nameTag:
        print('id:' + j['id'] + ', name:' + j.string)
        tel = i.find('tel')
        print('tel : ' + tel.string)
    for j in i.find_all('exam'):
        print('kor:' + j['kor'] + ', eng:' + j['eng'])
    print()