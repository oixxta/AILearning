"""
기온에 따른 가게 매출액의 평균 차이 검정.
기온은 3개의 집단 : 더움, 보통, 추움으로 분류

귀무가설 : 기온에 따른 가게 매출액 평균의 차이는 없다.(세 집단은 독립적임)
대립가설 : 기온에 따른 게게 매출액 평균의 차이는 있다.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

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

# 데이터를 온도별로 3개의 집단으로 분리 : 일별 최고온도(연속형 자료) 변수를 이용해서 명목형으로 변환
print(data.maxTa.describe())    #최고 : 36도, 최저 : -4.9도, 평균 : 18.5도
data['ta_index'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=[0, 1, 2]) #1구간 : -5 ~ 8, 2구간 : 8 ~ 24, 3구간 : 24 ~ 37
print(data.head(3))
print(data.ta_index.unique())   #[2, 1, 0]
print(data.isnull().sum())      #null 인자 없음을 확인.

# 최고온도를 세 그룹으로 나눈 후, 등분산/정규성 검정
x1 = np.array(data[data.ta_index == 0].AMT)
x2 = np.array(data[data.ta_index == 1].AMT)
x3 = np.array(data[data.ta_index == 2].AMT)
print(x1[:5], x2[:5], x3[:5])
print(stats.levene(x1,x2,x3).pvalue)    #0.039002396565063324, 등분산 불만족.
print(stats.shapiro(x1).pvalue) #0.2481924204382751, 정규성 만족
print(stats.shapiro(x2).pvalue) #0.03882572120522948, 정규성 불만족
print(stats.shapiro(x3).pvalue) #0.3182989573650957, 정규성 만족. 3개 중 2개가 만족함으로 정규성 만족으로 간주

# 온도별 매출액 평균
spp = data.loc[:,['AMT', 'ta_index']]
print(spp.groupby('ta_index').mean())   #과학적 표기법으로 나옴.
print(pd.pivot_table(spp, index=['ta_index'], aggfunc='mean'))

# ANOVA 분석 : f_oneway
sp = np.array(spp)
group1 = sp[sp[:, 1] == 0, 0]
group2 = sp[sp[:, 1] == 1, 0]
group3 = sp[sp[:, 1] == 2, 0]

print(stats.f_oneway(group1, group2, group3))   #pvalue=np.float64(2.360737101089604e-34), 거의 0에 가까움
#결과 해석 : 귀무가설 기각, 대립가설 채택. 기온에 따른 게게 매출액 평균의 차이는 있다.


# 참고 : 등분산성을 만족하지 않을 경우, Welch's anova test을 실시함., pip install pingouin
from pingouin import welch_anova
print(welch_anova(dv='AMT', between='ta_index', data=data)) #p-unc : 7.907874e-35
#결과 해석 : 귀무가설 기각, 대립가설 채택. 기온에 따른 게게 매출액 평균의 차이는 있다.


# 참고 : 정규성을 만족하지 않을 경우, Kruskal wallis test를 실시함.
print(stats.kruskal(group1, group2, group3))    #pvalue=np.float64(1.5278142583114522e-29))
#결과 해석 : 귀무가설 기각, 대립가설 채택. 기온에 따른 게게 매출액 평균의 차이는 있다.


# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukeyResult = pairwise_tukeyhsd(endog=spp['AMT'], groups=spp['ta_index'])
print(tukeyResult)

tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()