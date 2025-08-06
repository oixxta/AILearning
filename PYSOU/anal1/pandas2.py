import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

# 재색인
# Series의 재색인
data = Series([1, 3, 2], index=(1, 4, 2))
print(data)
data2 = data.reindex((1, 2, 4)) #1, 2, 4 순으로 재 인덱싱
print(data2)

# 재색인 할 때 값 채우기
data3 = data2.reindex([0, 1, 2, 3, 4, 5])
print(data3)    #없는 값들은 모두 NaN으로 뜸
# 대응값이 없는(NaN) 인덱스는 결측값인데 777로 채우기
data3 = data2.reindex([0, 1, 2, 3, 4, 5], fill_value=777)
print(data3)
# 결측값이 없을경우 이전 값으로 다음값을 채움
print()
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='ffill')   #method='pad' 로도 가능!
print(data3)
# 결측값이 없을경우 앞의 값으로 다음값을 채움
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='bfill')   #method='backfill' 로도 가능!
print(data3)

# bool 처리, 슬라이싱 관련 method : loc(), iloc()
df = DataFrame(np.arange(12).reshape(4, 3), index=['1월', '2월', '3월', '4월'],
               columns=['강남', '강북', '서초'])            #4행3열짜리 데이터 생성
print(df)
print(df['강남'])       #강남 열만 출력
print(df['강남'] > 3)   #강남 열 중 3을 초과하는 것만
print(df[df['강남'] > 3])   #강남 중 데이터 값이 3을 초과하는 것만
print()
# 복수 인덱싱 : loc는 라벨을 지원함, iloc는 숫자를 지원함.
print(df.loc[:'2월'])   #2월 이하의 것들만 출력
print(df.loc[:'2월', ['서초']]) #2월 이하 행의 서초만 출력
print()
print(df.iloc[2])   #2행의 내용 출력
print(df.iloc[2, :])
print(df.iloc[:3])
print(df.iloc[:3, 2], type(df.iloc[:3, 2]))   #3행 2열의 내용 출력
print(df.iloc[:3, 1:3])