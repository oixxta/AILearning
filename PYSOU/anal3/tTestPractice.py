import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon
import sys
import pickle
import MySQLdb


"""
one-sample t 검정 : 문제 1)
영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
305 280 296 313 287 240 259 266 318 280 325 295 315 278

귀무가설 : 백열전구의 수명은 300시간이다.
대립가설 : 백열전구의 수명은 300시간이 아니다. (더 크거나, 더 작거나)
"""
def tTestPractice1():
    data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
    dataMean = np.array(data).mean()    #평균값 : 289.7857142857143
    print(dataMean)
    # ttest_1samp를 사용한 검정
    result1 = stats.ttest_1samp(data, popmean=300)
    print('statistic:%.5f, pvalue:%.5f'%result1)    #statistic:-1.55644, pvalue:0.14361
    #결론 : p값이 0.05보다 크기 때문에 귀무가설 채택
    #백열전구의 수명은 300시간이다.

"""
one-sample t 검정 : 문제2)
국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 
A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
실습 파일 : one_sample.csv
참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")

귀무가설 : 노트북 평균 사용 시간은 5.2 시간이다.
대립가설 : 노트북 평균 사용 시간이 5.2 시간이 아니다.
"""
def tTestPractice2():
    data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv")
    data.time.replace(("     ", ""))    # time의 값들 중 '     '으로 지정된 것들을 빈 공간으로 만들기
    data['time'] = pd.to_numeric(data['time'], errors='coerce') # time의 값들을 string형에서 int형으로 바꾸기
    data = data.dropna(subset=['time'])    #time 값들 중 결측치들 제거
    print(data.head(10))
    print(data.describe())  #데이터 요약 통계량

    #정규성 검정 : one-sample t-test는 옵션. 필수는 아님.
    print(stats.shapiro(data.time)) #pvalue=np.float64(0.7242303336695732), p값이 0.05보다 큼으로 정규분포를 충족함.

    #wilcoxon를 사용한 검정
    wilcox_result = wilcoxon(data.time - 5.2)
    print('wilcox_result : ', wilcox_result)    #pvalue=np.float64(0.00025724108542552436)

    #결론 : p값이 0.05보다 훨씬 작기 때문에 귀무가설 기각, 대립가설 채택
    #노트북 평균 사용 시간이 5.2 시간이 아니다.
    
"""
one-sample t 검정 : 문제3)
https://www.price.go.kr/tprice/portal/main/main.do 에서 
메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

귀무가설 : 전국 평균 미용 요금이 15000원이다.
대립가설 : 전국 평균 미용 요금이 15000원이 아니다.
"""
def tTestPractice3():
    data = pd.read_csv("./haircut.csv")
    
    data['price'] = pd.to_numeric(data['price'], errors='coerce')    # price 값들을 string형에서 int형으로 바꾸기
    data = data.dropna()    #결측치(세종특별자치시의 price) 제거
    print(data)
    print(data.describe())    #데이터 요약 통계량, mean : 19512.875000

    #ttest_1samp를 이용한 검정
    result = stats.ttest_1samp(data.price, popmean=15000)
    print(result)           #pvalue=np.float64(7.410669455004333e-06)
    #wilcoxon를 사용한 검정
    result2 = wilcoxon(data.price - 15000)
    print(result2)          #pvalue=np.float64(3.0517578125e-05)

    #결론 : p값이 0.05보다 훨씬 작기 때문에 귀무가설 기각, 대립가설 채택
    #전국 평균 미용 요금이 15000원이 아니다.

"""
two-sample t 검정 : 문제1)
다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.

귀무가설 : 포장지 색상에 따른 제품의 매출액 차이는 없다.
대립가설 : 포장지 색상에 따른 제품의 매출액 차이는 있다.
"""
def tTestPractice4():
    #1. 데이터 가져오기
    blueData = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
    blue = pd.DataFrame(blueData, columns=['sales'])
    print(blue.head(3))
    print('블평 : ', blue.mean())
    redData = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]
    red = pd.DataFrame(redData, columns=['sales'])
    print(red.head(3))
    print('레평 : ', red.mean())

    #2. 데이터 가공하기(필요없읍)

    #3. 데이터들의 정규성 검정
    print(stats.shapiro(red).pvalue)    #0.5347933246260025, 0.05보다 큼으로 정규성 만족
    print(stats.shapiro(blue).pvalue)   #0.5102310078114559, 0.05보다 큼으로 정규성 만족

    #4. 두 데이터 사이의 등분산성 검정
    print(stats.levene(blue, red).pvalue)   #0.43916445, 0.05보다 큼으로 등분산성 만족

    #5. 등분산성 가정이 적절하기 때문에, equal_var=True로 독립 t-Test
    print(stats.ttest_ind(red, blue, equal_var=True))   #pvalue=array([0.00831655])
    #결과 해석 : p값이 0.05보다 작기 때문에, 대립가설 채택.
    #포장지 색상에 따른 제품의 매출액 차이는 있다.

"""
two-sample t 검정 : 문제2)
아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여
혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.

귀무가설 : 남녀의 혈관 내의 콜레스테롤 양은 차이가 없다.
대립가설 : 남녀의 혈관 내의 콜레스테롤 양은 차이가 있다.
"""
def tTestPractice5():
    #1. 데이터 가져오기
    maleData = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
    maleDf = pd.DataFrame(data=maleData, columns=['colCount'])
    #print(maleDf.head(3))
    femaleData = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]
    femaleDf = pd.DataFrame(data=femaleData, columns=['colCount'])
    #print(femaleDf.head(3))

    #2. 데이터 가공하기
    filteredMale = maleDf.sample(n = 15, random_state = 1)  #15개 랜덤 추출, 시드넘버 1로 고정
    #print(filteredMale)
    filteredFemale = femaleDf.sample(n = 15, random_state = 1)
    #print(filteredFemale)

    #3. 데이터들의 정규성 검정
    print(stats.shapiro(filteredMale).pvalue)   #p : 0.5887004990706299, 0.05보다 큼으로 정규성 만족
    print(stats.shapiro(filteredFemale).pvalue) #p : 0.0034335619900264535, 0.05보다 작음으로 정규성 불만족

    #4. 두 데이터 사이의 등분산성 검정
    print(stats.levene(filteredMale, filteredFemale).pvalue)  #p: 0.58206502, 0.05보다 큼으로 등분산성 만족

    #5. 한쪽의 정규성이 불만족이기에. equal_var=False로 독립 t-Test
    print(stats.ttest_ind(filteredMale, filteredFemale, equal_var=False)) #pvalue=array([0.26561264]), 0.05보다 큼으로 정규성 만족

    #결과 해석 : p값이 0.05보다 크기 때문에, 귀무가설 채택.
    #남녀의 혈관 내의 콜레스테롤 양은 차이가 없다.

"""
two-sample t 검정 : 문제3)
DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

귀무가설 : 총무부, 영업부 직원의 연봉의 평균에 차이는 없다. (서로 독립)
대립가설 : 총무부, 영업부 직원의 연봉의 평균에 차이가 있다.
"""
def tTestPractice6():
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
            SELECT busernum, jikwonpay
            FROM jikwon
        """
        cursor.execute(sql)
        dfFromMaria = pd.DataFrame(cursor.fetchall(), columns=['busernum', 'jikwonpay'])
        print(dfFromMaria)
    except Exception as e:
        print('error!', e)
    finally:
        cursor.close()
        conn.close()

    #2. 데이터 가공하기


    
    chongmuDf = dfFromMaria.loc[dfFromMaria['busernum'] == 10, :]   #총무부만 든 DF
    chongmuDf['jikwonpay'].fillna(chongmuDf['jikwonpay'].mean(), inplace=True)    #결측치에 평균값 삽입
    print(chongmuDf)
    yeongupDf = dfFromMaria.loc[dfFromMaria['busernum'] == 20, :]   #영업부만 든 DF
    yeongupDf['jikwonpay'].fillna(yeongupDf['jikwonpay'].mean(), inplace=True)    #결측치에 평균값 삽입
    print(yeongupDf)

    #3. 데이터들의 정규성 검정
    tg1 = chongmuDf['jikwonpay']
    tg2 = yeongupDf['jikwonpay']
    print('두 집단 평균 : ', np.mean(tg1), ' ', np.mean(tg2))   #5414.285714285715   4908.333333333333
    print(stats.shapiro(tg1).pvalue)    #p : 0.026044936412817302, 0.05보다 작음으로 정규성 불만족함.
    print(stats.shapiro(tg2).pvalue)    #p : 0.025608399511523605, 0.05보다 작음으로 정규성 불만족함.

    #4. 두 데이터 사이의 등분산성 검정
    #두 데이터 다 정규성을 불만족하기 때문에 스킵

    #5. equal_var=False로 독립 t-Test
    print(stats.ttest_ind(tg1, tg2, equal_var=False))  #pvalue=np.float64(0.6668004418280253)
    #결과 해석 : p값이 0.05보다 크기 때문에, 귀무가설 채택.
    #총무부, 영업부 직원의 연봉의 평균에 차이는 없다.


#메인메서드
#tTestPractice1()
#tTestPractice2()
#tTestPractice3()
#tTestPractice4()
#tTestPractice5()
tTestPractice6()
