"""
세개 이상의 모집단에 대한 가설검정 : 분산분석(ANOVA)

분산분석이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어
요인에 의한 분산이 의미있는 크기를 가지는지 검정하는 것을 의미함.

세 집단 이상의 평균비교에서는 독립인 두 집단의 평균비교를 반복하여 실시할 경우에 제1종 오류가
증가하게 되어 문제가 발생함.

이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.

분산의 성질과 원리를 이용하여, 평균의 차이를 분석한다.
즉, 평균을 직접 비교하지 않고 집단 내 분산과 집단간 분산을 이용하여 집단의 평균이 서로
다른지 확인하는 방법이다.

F-value = 그룹간 분산(between variance) / 그룹내 분산(within variance)
"""

"""
3개 이상 집단 평균 비교 시 절차
1) 정규성 + 등분산성 가정 충족 여부
    - 정규성 만족 + 등분산성 만족 → 일원분산분석(One-way ANOVA)
    - 정규성 만족 + 등분산성 불만족 → Welch’s ANOVA
    - 정규성 불만족 → 비모수 검정(Kruskal-Wallis test)

2) 사후검정(Post-hoc test)
    - ANOVA는 "차이가 있다"까지만 알려주므로, 차이가 나는 구체적인 집단쌍을 알고 싶으면 사후검정을 추가해야 한다.
    - 정규성 만족 + 등분산성 만족 → Tukey’s HSD test
    - 등분산성 불만족 → Games-Howell test
    - 비모수일 때 → Dunn’s test, Conover’s test 등 사용
"""

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


"""
실습 1 ) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시.
three_samples.csv

독립변수 : 교육방법 (세가지 방법), 종속변수 : 시험점수
Oneway ANOVA 사용 - 복수의 집단을 대상으로 집단을 구분하는 요인이 하나일 때.

귀무가설 : 교육방법간의 시험점수의 차이는 없다.
대립가설 : 교육방법간의 시험점수의 차이는 있다.
"""
#데이터 가져오기 및 데이터 확인
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv")
print(data.head(3))
print(data.shape)
print(data.describe())  #score의 max값이 500임 : 의심자료

#이상치를 차트로 확인
#plt.hist(data.score)
#plt.boxplot(data.score) #두 개의 이상치를 확인 (100점을 넘어간 값들)
#plt.show()
#plt.close()

#이상치 제거 및 데이터 가공
data = data.query('score <= 100')
print(len(data))

result = data[['method', 'score']]
print(result)
m1 = result[result['method'] == 1]  #1번 교육방식으로 배운 애들
m2 = result[result['method'] == 2]  #2번 교육방식으로 배운 애들
m3 = result[result['method'] == 3]  #3번 교육방식으로 배운 애들
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']

#정규성 검토 : score에 대한 정규성과 등분산성 검토 필요
print('score1 : ', stats.shapiro(score1).pvalue)    #0.17467355591727662, 정규성 만족
print('score2 : ', stats.shapiro(score2).pvalue)    #0.3319001150712364, 정규성 만족
print('score3 : ', stats.shapiro(score3).pvalue)    #0.11558564512681252, 정규성 만족

print(stats.ks_2samp(score1, score2))   #한번에 두 집단의 동일분포 여부를 하는 방법

#등분산성 검토(복수 집단의 분산의 치우침 정도)
print(stats.levene(score1, score2, score3).pvalue)      #0.11322850654055751, 등분산성 만족
print(stats.fligner(score1, score2, score3).pvalue)     #0.10847180733221087, 등분산성 만족
print(stats.bartlett(score1, score2, score3).pvalue)    #0.10550176504752282, 등분산성 만족, 비모수 검정때만 사용함.

#ANOVA 실시 : 정규성 만족 + 등분산성 만족 → 일원분산분석(One-way ANOVA)
reg = ols("data['score'] ~ C(data['method'])", data = data).fit() #단일회귀 모델 제작. 종속변수, 독립변수 순으로(독립변수가 범주형일경우 C를 붙여야 함.)
table = sm.stats.anova_lm(reg, type=2)  #type=2가 기본값, 분산 분석표를 이용해 분산결과 작성함.
print(table)    #F value : 0.062312, p value : 0.939639, 귀무가설 채택.
#결론 : 교육방법간의 시험점수의 차이는 없다.

#사후검정 실시 : ANOVA에선 사후검정이 필수임.
#분산분석은 집단의 평균에 차이 여부만 알려줄 뿐, 각 집단 간의 평균 차이는 알려주지 않는다.
#각 집단 간의 평균 차이를 확인하기 위해 사후검정 실시.
turResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(turResult)
"""
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
     1      2   0.9725 0.9702 -8.9458 10.8909  False
     1      3   1.4904 0.9363 -8.8183  11.799  False
     2      3   0.5179 0.9918 -9.6125 10.6483  False
----------------------------------------------------
"""
#사후검정 시각화
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()