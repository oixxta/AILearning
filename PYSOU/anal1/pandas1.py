# pandas  : 행과 열이 단순 정수형 인덱스가 아닌, 레이블로 식별되는 numpy의 구조화된 배열을 
#           보완(강)한 모듈임. 고수준의 지료구조(시계열 축약연산, 누락데이터 처리, SQL, 시계화 등을 지원)

import pandas as pd
from pandas import Series
from pandas import DataFrame

import numpy as np


#Series : 일련의 객체를 담을 수 있는 1차원 배열과 유사한 자료구조로 색인을 갖음
# list, array로 부터 만들 수 있음.

obj = Series([3, 7, -5, 4]) #list, tuple 가능, set 불가(set은 순서가 없기 때문에 인덱싱 불가)
print(obj, type(obj))       #자동 인덱스(명시적)
print(obj[1])
obj2 = Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])    #기존 숫자 대신, 문자로 인덱스 지정
print(obj2['b'])
print(obj2.sum(), np.sum(obj2))     #파이썬 제공 sum, 넘파이 제공 sum

print(obj2.values)  #값들만 출력하는 명령어
print(obj2.index)   #인덱스들만 출력하는 명령어


#판다스의 슬라이싱
print(obj2['a'])    #인덱스 값이 'a'인 값 출력
print(obj2[['a']])  #인덱스 값과 내용물 값 동시 출력
print(obj2[['a', 'b']]) #두개 출력
print(obj2['a' :'c'])   #범위 출력
#print(obj2[3])      #위에서 커스텀 인덱스값을 지정했기에, 숫자 인덱스로는 오류가 발생함!
print(obj2.iloc[3])  #iloc[]를 쓰면 커스텀 인덱스값이 지정되었어도 숫자로도 호출됨.
print(obj2.iloc[[2, 3]])
print(obj2 > 0)     #obj2의 값들 중 0보다 클 경우 True 출력
print('a' in obj2)  #obj2의 인덱스들 중 'a'가 존재하는 지 여부에 대해 출력

print('\ndict type으로 Series 객체 생성')
names = {'mouse' : 5000, 'keyboard' : 25000, 'monitor' : 450000}    #dict type으로 Series 객체 생성
print(names, type(names))
obj3 = Series(names)
print(obj3, type(obj3)) #dict타입의 key가 자동으로 인덱스로 지정, 내용물도 자동으로 값으로 지정
obj3.index = ['마우스', '키보드', '모니터'] #이미 지정된 인덱스를 바꿀 수 있음.
print(obj3)
print(obj3['마우스'])   #인덱스 값으로 값 호출 가능
print(obj3[0])  #기능 오류 발생

obj3.name = '상품가격'  #obj3의 Series 이름을 '상품가격'으로 지정.
print(obj3)

print('-----------------------------------')
#데이터 프레임 : Series 객체가 모여 표를 구성.
df = DataFrame(obj3)
print(df)

data = {    #dict type
    'irum' : ['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'juso' : ['역삼동', '신당동', '역삼동', '역삼동', '신당동'],
    'nai' : [23, 25, 33, 30, 35]
}
frame = DataFrame(data)         #dict를 이용한 데이터프레임 만들기(표의 형태로)
print(frame)

print(frame['irum'])    #위에서 만든 표에서 칼럼 얻기 1
print(frame.irum)       #위에서 만든 표에서 칼럼 얻기 2
print(type(frame.irum)) #데이터 프레임에서 뽑아낸 열의 타입은 Series 타입이 됨!

print(DataFrame(data, columns=['juso', 'irum', 'nai'])) #열 순서는 바꾸는게 가능함(별로 중요하진 않음)

print('\ndata에 NaN(데이터 없음)을 넣기')
frame2 = DataFrame(data, columns=['irum', 'juso', 'nai', 'tel'],
                   index=['a', 'b', 'c', 'd', 'e'])
print(frame2)       #tel 열의 모든 값은 NaN

#NaN으로 된 값들에 새 값을 넣기
frame2['tel'] = '111-1111'      #모든  tel 열에 같은 값 넣기
print(frame2)
val = Series(['222-2222', '333-3333', '444-4444'], index=['b', 'c', 'e'])   #특정 인덱스의 특정 값
frame2.tel = val
print(frame2)

print(frame2.T) #Transpose, 행과 열이 바뀜

print(frame2.values)    #2차원 배열 중복리스트 형식으로 반환
print(frame2.values, type(frame2.values))   #문자열로 취급되기에 ndarray 타입이 됨
print(frame2.values[0, 1]) #0행 1열의 값 출력
print(frame2.values[0:2])  #0행과 1행의 내용 출력

#행/열 삭제
frame3 = frame2.drop('d', axis=0)   #인덱스가 d인 행 삭제
print(frame3)
frame4 = frame2.drop('tel', axis=1) #열 이름이 tel인 열 삭제
print(frame4)

#정렬
print('-----------')
print(frame2.sort_index(axis=0, ascending=True))    #행단위 내림차순 정렬 결과
print(frame2.sort_index(axis=1, ascending=False))   #열단위 오름차순 정렬 결과

print(frame2['juso'].value_counts())    #'주소' 값들의 갯수 반환

#문자열 자르기
print('----------')
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
fr = pd.DataFrame(data)
print(fr)
result1 = Series([x.split()[0] for x in fr.juso])   #공백을 구분자로 문자열 분리
result2 = Series([x.split()[1] for x in fr.juso])   #공백을 구분자로 문자열 분리
print(result1, result1.value_counts())
print(result2, result2.value_counts())