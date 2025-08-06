import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#Pandas로 파일 저장
items = {'apple':{'count': 10, 'price': 1500},
         'orange':{'count': 4, 'price': 700}
         }
df = DataFrame(items)
print(df)

#df.to_clipboard()
#print(df.to_html())
#print(df.to_json())
df.to_csv('result.csv', sep = ',', index=False, header=False)

data = df.T
print(data)
data.to_csv('result.csv', sep = ',', index=False, header=True)


#엑셀 관련 파일 입출력
df2 = DataFrame({'name':['Alice', 'Bob', 'Oscar'], 'age':[24, 26, 33], 'city':['seoul', 'suwon', 'incheon']})
print(df2)
df2.to_excel('result2.xlsx', index = False, sheet_name='mysheet')   #저장, 엑셀에는 sheet가 있기 때문에 시트의 이름을 지정해줘야함!
exdf = pd.ExcelFile('result2.xlsx') #읽기
print(exdf.sheet_names)
dfexcel = exdf.parse('mysheet')
print(dfexcel)
