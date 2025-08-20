"""
비(눈) 여부에 따른 가게 매출액의 평균 차이 검정
공통 칼럼이 년월일인 두 개의 파일(날씨 파일, 매출액 파일)을 조합해서 작업(두 개의 집단을 사용함)

귀무가설 : 강수량에 따른 가게 매출액 평균의 차이는 없다.(두 집단은 독립적임)
대립가설 : 강수량에 따른 게게 매출액 평균의 차이는 있다.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출액 데이터 읽어오기
salesData = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv", dtype={'YMD':'object'})
print(salesData.head(3))
print(salesData.info())    # 328 rows, 3 columns. YMD:20180601

# 날씨자료 데이터 읽어오기
wearherData = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv")
print(wearherData.head(3))
print(wearherData.info())  # 702 rows, 9 columns. tm:2018-06-01, 날짜의 타입을 바꿔야함.

# salesData 데이터의 날짜를 기준으로 두 데이터를 병합하기
wearherData.tm = wearherData.tm.map(lambda x:x.replace('-', ''))   #먼저 날짜의 모양부터 맞춰야 함.
print(wearherData.head(1))
myDf = salesData.merge(wearherData, how='left', left_on='YMD', right_on='tm')   #레프트 조인으로 머지
print(myDf)
print(len(myDf))
print(myDf.columns)
data = myDf.iloc[:, [0,1,7,8]]  #myDf 전체 행 중 날짜, 매출액, 최고기온, 강수량 데이터를 긁어와 data에 저장
print(data.head(3))
print(data.isnull().sum())  #결측치 갯수 확인

# 강수 여부에 따른 매출액 평균 비교 : 차이가 유의미한지 확인하기
data['rain_yn'] = (data['sumRn'] > 0).astype(int) # 새 칼럼을 만들어서 비오는 날 여부를 저장하는 방법 1, 비가 오면 1, 안오면 0
#data['rain_yn'] = (data.loc[:, ('sumRn')] > 0) * 1 # 새 칼럼을 만들어서 비오는 날 여부를 저장하는 방법 2, 비가 오면 1, 안오면 0
print(data.head(5))

sp = np.array(data.iloc[:, [1,4]])  #AMT와 sumR만 저장된 새 데이터프레임 만들기
tg1 = sp[sp[:, 1] == 0, 0]  #그룹 1 : 비 안오는 날의 매출액들
tg2 = sp[sp[:, 1] == 1, 0]  #그룹 2 : 비 오는 날의 매출액들
print(tg1[:3])
print(tg2[:3])

import matplotlib.pyplot as plt
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

print('두 집단 평균 : ', np.mean(tg1), ' ', np.mean(tg2))   #761040.2542372881   757331.5217391305

#정규성 검정 시작
print(len(tg1), len(tg2))
print(stats.shapiro(tg1).pvalue)    # p : 0.056050644029515644, 0.05보다 큼으로 정규성 만족함.
print(stats.shapiro(tg2).pvalue)    # p : 0.8827503155277691, 0.05보다 큼으로 정규성 만족함.

#등분산성 검정 시작
print('등분산성 : ', stats.levene(tg1, tg2).pvalue) # 0.7123452333011173, 0.05보다 큼으로 정규성 만족함.

#등분산성 가정이 적절하기 때문에, equal_var=True로.
print(stats.ttest_ind(tg1, tg2, equal_var=True))   #pvalue=np.float64(0.919534587722196)
# 결과 : p값이 0.05보다 크기 때문에, 귀무가설 채택.
# 강수량에 따른 가게 매출액 평균의 차이는 없다.(두 집단은 독립적임)

