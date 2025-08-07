# JSON : XML에 비해 가벼우며, 배열에 대한 지식만 있으면 처리 가능함.

import json

dict = {'name':'tom', 'age':33, 'score':['90', '80', '100']}
print("dict:%s"%dict)
print(type(dict))

print('json Incoding (dict to json shaped string)---')
str_val = json.dumps(dict)  #딕트를 문자열로 바꿀 때 사용
print("str_val%s"%str_val)
print(type(str_val))
#print(str_val['name'])  TypeError: string indices must be integers, not 'str'

print('json decoding (str to dict shaped string)---')
json_val = json.loads(str_val)
print("json_val%s"%json_val)
print(type(json_val))
print(json_val['name'])

for k in json_val.keys():
    print(k)

print('웹에서 JSON 문서 읽기 -----------------')
import urllib.request as req
url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"
plainText = req.urlopen(url).read().decode()
print(plainText)    #string 타입임, dict로 가공 필요.
jsonData = json.loads(plainText)    #dict 타입으로 가공함
print(type(jsonData))
print(jsonData['SeoulLibraryTime']['row'][0]['LBRRY_NAME']) #JSON 형식의 특정 데이터 접근

# dict의 자료를 읽어 도서관명, 전화, 주소를 출력하고 데이터 프레임에 넣기
libData = jsonData.get('SeoulLibraryTime').get('row')
#print(libData)
print(libData[0].get('LBRRY_NAME'))     #LH강남3단지작은도서관

datas = []

for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    #print(name + '\t' + tel + '\t' + addr)
    datas.append([name, tel, addr])

import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명', '전화번호', '주소'])
print(df)


