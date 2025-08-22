"""
추론통계 분석 중 비율검정- 비율검정 특징
: 집단의 비율이 어떤 특정한 값과 같은지를 검증.
: 비율 차이 검정 통계량을 바탕으로 귀무가설의 기각여부를 결정.
"""
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd
import scipy.stats as stats

"""
# one-sample
A회사에는100명 중에45명이 흡연을 한다.국가통계를 보니 국민 흡연율은 35%라고 한다.
비율이 같냐?

귀무가설 : A회사 직원들의 흡연율과 국민 평균 흡연율은 같다.
대립가설 : A회사 직원들의 흡연율과 국민 평균 흡연율은 다르다.
"""
count = np.array([45])
nobs = np.array([100])
val = 0.35
z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)
print(p)    #0.04442318
#결론 : 귀무가설 기각, A회사 직원들의 흡연율과 국민 평균 흡연율은 다르다.(비율이 다르다)


"""
# two-sample
A회사 사람들 300명 중 100명이 커피를 마시고,B회사 사람들 400명 중 170명이 커피를 마셨다.
비율이 같냐?

귀무가설 : 
대립가설 : 
"""
count = np.array([100, 170])
nobs = np.array([300, 400])
z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)
print(p)    #1.5028294294082938e-32
#결론 : 귀무가설 기각, 비율이 다르다.


"""
이항분포, 이항검정
'결과가 두 가지 값을 가지는 확률변수의 분포를 판단' 하는데 효과적.
예) 10명의 자격증 시험 합격자 중 여성이 6명, 남성이 4명이었다고 할 때, '여성이 남성보다 합격률이
높다'라고 할 수 있는가?
"""

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv")
print(data)
ctab = pd.crosstab(index=data['survey'], columns='count')
ctab.index = ['불만족', '만족']     #불만족 : 14, 만족 : 136
print(ctab)
#귀무가설 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80%이다.
#대립가설 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80%가 아니다.

#양측 검정 : 방향성이 없다.
x = stats.binomtest(k=136, n=150, p=0.8, alternative='two-sided')
print(x.pvalue) #0.0006734701362867024
#결론 : 귀무 기각, 대립가설 채택


#단측 검정 : 방향성이 있다. (80%보다 크다고 가정하고 검증)
result = stats.binomtest(k=136, n=150, p=0.8, alternative='greater')    #작을경우, less로 쓰기
print(result.pvalue)    #0.00031794019219854805
#결론 : 귀무 기각, 대립가설 채택
