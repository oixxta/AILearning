import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

#판다스의 연산

#시리즈 연산
s1 = Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = Series([4, 5, 6, 7], index = ['a', 'b', 'd', 'c'])
print(s1)
print(s2)
print(s1 + s2)      #D는 결측치로 들어감
print(s1.add(s2))   #위와 같음
print(s1.multiply(s2))  #곱하기, sub, div도 존재
print()
#데이터프레임 연산
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['seoul', 'daejeon', 'daegu'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['seoul', 'daejeon', 'jeju', 'suwon'])
print(df1)
print(df2)
print(df1 + df2)    #대응되는 것만 연산, 나머지는 결측치
print(df1.add(df2, fill_value=0))   #위와 같은 연산, 존재하지 않는 값들은 fill_value값으로 채우고 연산.

ser1 = df1.iloc[0]
print(ser1) #데이터프레임의 0행 값을 시리즈화
print(df1 - ser1)   #Broadcasting 연산

#결측치, 기술적 통계 관련 함수 & 매서드
print('결측치 관련 함수 & 매서드')
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.nan, None], [0.5, -1]], columns=['one', 'two'])   #4행 2열 데이터프레임
print(df)   #결측값들
print(df.isnull())  #널값 여부들 확인, bool형으로
print(df.notnull())  #널값이 아닌 여부들 확인, bool형으로
print(df.drop(0))   #데이터프레임의 0행을 삭제함
print(df.dropna()) #결측치(NaN)가 단 하나라도 들어있는 모든 행을 삭제함.
print(df.dropna(how='any')) #
print(df.dropna(how='all')) #
print(df.dropna(subset=['one'])) #one으로 이름이 지정된 열의 nan을 모두 지움
print(df.dropna(axis='rows'))       #NaN이 포함된 행 삭제
print(df.dropna(axis='columns'))    #NaN이 포함된 열 삭제
print(df.fillna(0))     #NaN 값들을 0으로 채움(0 외에 평균, 특정값, 이전값, 다음값도 가능함!)

print('기술통계 관련 함수 & 매서드')
print(df.sum()) #열의 합 - NaN은 연산에서 제외
print(df.sum(axis=0))   #
print(df.sum(axis=1))   #
print(df.describe())    #요약 통계량 전체 출력(자주 사용됨)
print(df.info())        #데이터프레임의 구조를 출력

#
print('재구조화, 구간 설정, agg 함수')
df = DataFrame(1000 + np.arange(6).reshape(2, 3), index = ['서울', '대전'], columns=['2020', '2021', '2022'])
print(df)
print(df.T) #재구조화
#Stack, unStack
df_row = df.stack() #열 -> 행으로 변경. 열 쌓기
print(df_row)
df_col = df_row.unstack()   #행 -> 열로 변경
print(df_col)

print()
#구간 설정 : 연속형 자료를 범주화
price = [10.3, 5.5, 7.8, 3.6]       
cut = [3, 7, 9, 11]     #구간 기준값
result_cut = pd.cut(price, cut)
print(result_cut)
print(pd.value_counts(result_cut))

datas = pd.Series(np.arange(1, 1001))
print(datas.head(3))
print(datas.tail(3))
result_cut2 = pd.qcut(datas, 3)   #범주화, 3개의 구간으로 나눔.
print(result_cut2)
print(pd.value_counts(result_cut2)) #데이터 분포 확인


#그룹별 처리
print('-' * 50)
group_col = datas.groupby(result_cut2)
#print(group_col)
print(group_col.agg(['count', 'mean', 'std', 'min']))    #그룹별 소계 출력

def myFunc(groupData):  #함수 직접 정의
    return{
        'count':groupData.count(),
        'mean':groupData.mean(),
        'std':groupData.std(),
        'min':groupData.min()
    }
print(group_col.apply(myFunc))  #어플라이는 함수를 실행하는 함수
print(group_col.apply(myFunc).unstack) 