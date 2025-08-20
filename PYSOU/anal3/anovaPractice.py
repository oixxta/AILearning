import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import sys
import pickle
import MySQLdb
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

"""
[ANOVA 예제 1]
빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.

조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

귀무가설 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.
대립가설 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 있다.
"""
def anovaPractice1():
    #1. 데이터 가져오기
    data = pd.read_csv("oilBread.csv")
    data['quantity'].fillna(data['quantity'].mean(), inplace=True)  #결측치 평균값으로 채우기
    print(data)

    #2. 데이터 가공하기
    x1 = np.array(data[data.kind == 1].quantity)   #1번 기름빵
    x2 = np.array(data[data.kind == 2].quantity)   #2번 기름빵
    x3 = np.array(data[data.kind == 3].quantity)   #3번 기름빵
    x4 = np.array(data[data.kind == 4].quantity)   #4번 기름빵
    #print(x1)

    #3. 정규성 검토
    print(stats.shapiro(x1).pvalue) #0.8680405840743664, 정규성 만족
    print(stats.shapiro(x2).pvalue) #0.5923924912154501, 정규성 만족
    print(stats.shapiro(x3).pvalue) #0.48601083943678747, 정규성 만족
    print(stats.shapiro(x4).pvalue) #0.4162161718602888, 정규성 만족

    #4. 등분산성 검토
    print(stats.levene(x1,x2,x3,x4).pvalue) #0.3268969935062273, 등분산성 만족

    #5. ANOVA 검정 : 정규성과 등분산성 모두 만족, f_oneway test 실시.
    print(stats.f_oneway(x1,x2,x3,x4).pvalue)   #0.8482436666841788
    #결과 해석 : 귀무가설 채택, 름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.

    #6. 사후검정
    tukeyResult = pairwise_tukeyhsd(endog=data['quantity'], groups=data['kind'])
    print(tukeyResult)

    tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
    plt.show()
    plt.close()


"""
[ANOVA 예제 2]

DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 
연봉의 평균에 차이가 있는지 검정하시오. 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

귀무가설 : 부서별 연봉의 차이가 없다.
대립가설 : 부서별 연봉의 차이가 있다.
"""
def anovaPractice2():
    #1. 데이터 가져오기
    try :
        with open('myMaria.dat', mode='rb') as obj:
            config = pickle.load(obj)
    except Exception as e:
        print('fail to load DB...', e)
        sys.exit()
    
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        sql = """
            SELECT jikwonpay, buserno
            FROM jikwon INNER JOIN buser
            ON jikwon.busernum = buser.buserno
        """
        cursor.execute(sql)
        dfFromMaria = pd.DataFrame(cursor.fetchall(), columns=['jikwonpay', 'buserno'])
        #print(dfFromMaria)
    except Exception as e:
        print('fail to connect!', e)
    finally:
        cursor.close()
        conn.close()
    
    #2. 데이터 가공하기
    dfFromMaria.dropna()    #연봉이 없는 직원이 있다면 작업에서 제외
    chongmuDf = dfFromMaria.loc[dfFromMaria['buserno'] == 10, :]   #총무부만 든 DF
    chongmuDf = chongmuDf['jikwonpay']
    print('총무부 평균월급 : ', chongmuDf.mean())   #5414.285714285715
    yeongupDf = dfFromMaria.loc[dfFromMaria['buserno'] == 20, :]   #영업부만 든 DF
    yeongupDf = yeongupDf['jikwonpay']
    print('영업부 평균월급 : ', yeongupDf.mean())   #4908.333333333333
    jeonsanDf = dfFromMaria.loc[dfFromMaria['buserno'] == 30, :]   #전산부만 든 DF
    jeonsanDf = jeonsanDf['jikwonpay']
    print('전산부 평균월급 : ', jeonsanDf.mean())   #5328.571428571428
    gwannriDf = dfFromMaria.loc[dfFromMaria['buserno'] == 40, :]   #관리부만 든 DF
    gwannriDf = gwannriDf['jikwonpay']
    print('관리부 평균월급 : ', gwannriDf.mean())   #6262.5

    #3. 정규성 검토
    print(stats.shapiro(chongmuDf).pvalue)  #0.026044936412817302, 정규성 불만족
    print(stats.shapiro(yeongupDf).pvalue)  #0.025608399511523605, 정규성 불만족
    print(stats.shapiro(jeonsanDf).pvalue)  #0.41940720517769636, 정규성 반족
    print(stats.shapiro(gwannriDf).pvalue)  #0.9078027897950541, 정규성 만족 
    #4개중 두 개만 만족함으로 정규성 불만족으로 간주

    #4. 등분산성 검토
    print(stats.levene(chongmuDf,yeongupDf,jeonsanDf,gwannriDf).pvalue) #0.7980753526275928, 등분산성 만족

    #5. ANOVA 검정 : 정규성을 만족하지 않을 경우, Kruskal wallis test를 실시.
    print(stats.kruskal(chongmuDf,yeongupDf,jeonsanDf,gwannriDf).pvalue)   #0.6433438752252654
    #결론 : 귀무가설 채택, 부서별 연봉의 차이가 없다.

    #6. 사후검정
    tukeyResult = pairwise_tukeyhsd(endog=dfFromMaria['jikwonpay'], groups=dfFromMaria['buserno'])
    print(tukeyResult)
    #    group1 group2  meandiff    p-adj    lower      upper   reject
    #--------------------------------------------------------------------
    #    10     20     -505.9524    0.9588 -3292.2114 2280.3066  False
    #    10     30      -85.7143    0.9998  -3217.199 3045.7705  False
    #    10     40      848.2143    0.9202 -2823.7771 4520.2056  False
    #    20     30      420.2381    0.9756 -2366.0209 3206.4971  False
    #    20     40     1354.1667    0.6937 -2028.2234 4736.5568  False
    #    30     40      933.9286    0.897 -2738.0628  4605.9199  False
    #----------------------------------------------------------------
    tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
    plt.show()
    plt.close()

anovaPractice1()
anovaPractice2()