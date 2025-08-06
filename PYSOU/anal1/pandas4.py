import numpy as np
import pandas as pd
from pandas import Series, DataFrame

print('-' * 50)
#데이터프레임 병합
df1 = DataFrame({'data1' : range(7), 'key' : ['b', 'b', 'b', 'c', 'a', 'a', 'b']})
print(df1)
df2 = DataFrame({'key' : ['a', 'b', 'd'], 'data2' : range(3)})
print(df2)                  #두 개의 데이터프레임 생성, 양쪽 모두 'key'칼럼을 가짐
print(pd.merge(df1, df2))   #머지 명령어로 두 개의 데이터프레임 병합, 머지를 하려면 기준키(공통열, 위 경우에는 key) 필수! (innerjoin, 교집합)
print(pd.merge(df1, df2, on='key')) # on = 은 생략 가능함
print(pd.merge(df1, df2, on='key', how='inner'))    # how=도 생략 가능, 이너조인
print(pd.merge(df1, df2, on='key', how='outer'))    # full 아우터 조인, 없는 값은 자동으로 결측치(NaN)화됨.
print(pd.merge(df1, df2, on='key', how='left'))     # left 아우터 조인
print(pd.merge(df1, df2, on='key', how='right'))    # right 아우터 

print('-' * 50)
#공통칼럼이 없을 시에 할 수 있는 데이터 병합 (머지 & 컨캣)
df3 = DataFrame({'key2' : ['a', 'b', 'c'], 'data2' : range(3)})
print(df3)  #df3은 df1, df2와 공통칼럼이 없는 데이터프레임!
print(pd.merge(df1, df3, left_on='key', right_on='key2'))   #left_on과 right_on 값을 지정해 머지 가능

print(pd.concat([df1, df3], axis=0))    #컨캣을 사용한 열 단위 결합
print(pd.concat([df1, df3], axis=1))    #컨캣을 사용한 행 단위 결합
s1 = Series([0, 1], index=['a','b'])
s2 = Series([2, 3, 4], index=['c','d', 'e'])
s3 = Series([5, 6], index=['f','g'])
print(pd.concat([s1, s2, s3], axis=0))      #컨켓은 시리즈도 병합 가능!

print('-' * 50)
#그룹화 : pivot_table, 데이터의 행렬 재구성 및 그룹화 연산
data = {'city':['gangnam', 'gangbuk', 'gangnam', 'gangbuk'],
        'year':[2000, 2001, 2002, 2003],
        'pop':[3.3, 2.5, 3.0, 2.0]
        }
df = DataFrame(data)
print(df)
print(df.pivot(index='city', columns='year', values='pop'))   #pivot으로 행과 열을 재구성, 없는 값은 결측치로 표현
print(df.set_index(['city', 'year']).unstack())               #set_index으로 행과 열을 재구성, 위와 같음
print(df.describe())        #요약 및 통계(4분위로)
print(df)
print(df.pivot_table(index=['city']))     #피벗 테이블 : 피벗과 그룹바이의 중간적 성격
print(df.pivot_table(index=['city'], aggfunc='mean'))   # aggfunc='mean'은 생략가능, 디폴트값, 위와 같은결과
print(df.pivot_table(index=['city', 'year'], aggfunc=[len, 'sum'])) #여러개의 매개변수도 가능함
print(df.pivot_table(index='city', values='pop', aggfunc='mean'))
print(df.pivot_table(index='city', values='pop', aggfunc='mean', margins=True)) #소계도 출력(ALL)
print(df.pivot_table(index='city', values='pop', aggfunc='mean', margins=True, fill_value=0))
hap = df.groupby(['city'])  #그룹바이
print(hap)
print(hap.sum())
print(df.groupby(['city']).sum())
print(df.groupby(['city', 'year']).mean())