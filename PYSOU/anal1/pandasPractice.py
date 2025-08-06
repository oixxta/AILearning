import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

#Pandas 문제1
def pandasQuOne():
    #표준정규분포를 따르는 9X4 형태의 DataFrame 생성
    myDf = DataFrame(np.random.randn(9, 4))
    #위에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정
    myDf.columns = ['No1', 'No2', 'No3', 'No4']
    print(myDf)
    #각 칼럼의 평균 구하기 : mean()함수와 axis 속성 사용
    print(myDf.mean(axis=0))


#Pandas 문제2
def pandasQuTwo():
    #DataFrame으로 위와 같은 자료를 만드시오.
    data = {
        'numbers' : [10, 20, 30, 40]
    }
    myDf = DataFrame(data, index=['a', 'b', 'c', 'd'])
    print(myDf)
    #c row의 값들을 가져오시오
    print('c row의 값 : ', myDf.loc['c'])
    #a, d row들의 값을 가져오시오.
    print('a, d row의 값 : ', myDf.loc['a'], myDf.loc['d'])
    #numbers의 합을 구하시오.
    print('numbers의 합 : ', myDf.sum())
    #numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.
    print(myDf * myDf)
    #floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5    아래 결과가 나와야 함.
    myDf.insert(1, 'floats', [1.5, 2.5, 3.5, 4.5])
    print(myDf)
    #names라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용.
    newCal = Series(['길동', '오정', '팔계', '오공'], index=['d', 'a', 'b', 'c'], name='names')
    myDf.insert(2, 'names', newCal)
    print(myDf)


#Pandas 문제3
def pandasQuThree():
    #5 x 3 형태의 랜덤 정수형 DataFrame을 생성하시오. (범위: 1 이상 20 이하, 난수)
    myDf = DataFrame(np.random.randint(1, 20, (5, 3)))
    print(myDf)
    #생성된 DataFrame의 컬럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.
    myDf.columns = ['A', 'B', 'C']
    myDf.index = ['r1', 'r2', 'r3', 'r4', 'r5']
    print(myDf)
    #A 컬럼의 값이 10보다 큰 행만 출력하시오.
    print('A 컬럼의 값이 10보다 큰 행')
    filteredDf = myDf[myDf['A'] > 10]
    print(filteredDf['A'])
    #새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.
    myDf['D'] = myDf['A'] + myDf['B']
    print(myDf)
    #행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.
    myDf = myDf.drop('r3', axis=0)
    print(myDf)
    #아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.
    myDf.loc['r6'] = {'A' : 15, 'B' : 10, 'C' : 2, 'D' : 25}
    print(myDf)


#Pandas 문제4
def pandasQuFour():
    #위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.
    data = {
        'product' : ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
        'price' : [12000, 25000, 150000, 900000],
        'stock' : [10, 5, 2, 3]
    }
    myDf = DataFrame(data, index=['p1', 'p2', 'p3', 'p4'])
    print(myDf)
    #price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.
    myDf['total'] = myDf['price'] * myDf['stock']
    print(myDf)
    #컬럼 이름을 다음과 같이 변경하시오. 원본 갱신
    myDf = myDf.rename(columns={'product' : '상품명', 'price' : '가격', 'stock' : '재고', 'total' : '총가격'})
    print(myDf)
    #재고(재고 컬럼)가 3 이하인 행의 정보를 추출하시오.
    print(myDf[myDf['재고'] < 3])
    #인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.
    print(myDf.loc['p2'])
    print(myDf.iloc[1])
    #인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오. (원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)
    newDf = myDf.drop('p3', axis=0)
    print(newDf)
    #위 DataFrame에 아래와 같은 행(p5)을 추가하시오.
    newDf.loc['p5'] = {'상품명' : 'USBmemory', '가격' : 15000, '재고' : 10, '총가격' : 15000 * 10}
    print(newDf)


#Pandas 문제5
def pandasQuFive():
    #타이타닉 csv 다운로드 후 데이터베이스화
    data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
    myDf = DataFrame(data)
    print(myDf)

    #1. 데이터프레임의 자료로 나이대에 대한 생존자 수 계산
    fillteredDf = myDf[myDf['Survived'] == 1]
    a = fillteredDf['Age']
    bins = [1, 20, 35, 60, 150]
    labels = ["소년", "청년", "장년", "노년"]
    result_cut = pd.cut(a, bins=bins, labels=labels)
    print(result_cut.value_counts())
    #2. 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다.
    




#Pandas 문제6


#pandasQuOne()
#pandasQuTwo()
#pandasQuThree()
#pandasQuFour()
pandasQuFive()