"""
집단간차이분석: 평균또는비율차이를분석
: 모집단에서추출한표본정보를이용하여모집단의다양한특성을과학적으로추론할수있다.

* T-test와 ANOVA의 차이- 두집단이하의변수에대한평균차이를검정할경우 
T-test를사용하여검정통계량T값을구해가설검정을한다. 
- 세집단이상의변수에대한평균차이를검정할경우에는ANOVA를이용하여검정통계량 F값을구해가설검정을한다
"""

# 핵심 아이디어 :
# 집단 평균차이(분자)와 집단 내 변동성(표준오차, 표준오차 등, 분모)을 비교하여,
# 그 차이가 데이터의 불확실성(변동성)에 비해 얼마나 큰 지를 계산한다.
# T 분포는 표본 평균을 이용해 정규분포의 평균을 해석할 때 많이 사용한다.

# 대개의 경우 표본의 크기는 30개 이하일 때 T 분포를 따른다.
# T검정은 '두개 이하 집단의 평균의 차이가 우연에 의한 것인지 통계적으로 유의한 차이를 판단하는 통계적 절차다.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

"""
예제 1) 어느 남성 집단의 평균키 검정

귀무가설 : 집단의 평균키는 177이다. (모수집단의 평균 키와 셈플데이터의 평균 키는 같다)
대립가설 : 집단의 평균키는 177이 아니다. (177보다 크거나, 오히려 작거나)
"""
oneSample = [167.0, 182.7, 160.6, 176.8, 185.0]
print(np.array(oneSample).mean())   #174.42
# 177.0과 174.42는 평균의 차이가 있느냐? (T test로 검증)
result = stats.ttest_1samp(oneSample, popmean=177)  #popmean : 모수의 평균값
print('statistic:%.5f, pvalue:%.5f'%result)   #statistic:-0.55499, pvalue:0.60847
#결과 : p값이 0.05보다 큼으로, 귀무가설을 채택함.
#집단의 평균키는 177이다. (모수집단의 평균 키와 셈플데이터의 평균 키는 같다), 더 정확하겐 이정도 차이는 평균키로 용인한다.

#만약 모수데이터의 평균값을 199로 높일 경우,
result = stats.ttest_1samp(oneSample, popmean=199)  #popmean : 모수의 평균값
print('statistic:%.5f, pvalue:%.5f'%result)     #statistic:-5.28750, pvalue:0.00614
#결과 : p값이 0.05보다 작음으로, 대립가설을 채택함.

#sns.displot(oneSample, bins=10, kde=True, color='blue')
#plt.xlabel('data')
#plt.ylabel('count')
#plt.show()
#plt.close()

"""
예제 2) 단일 모집단의 평균에 대한 가설검정(one samples t-test)
A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리(국어 점수 평균 검정)

귀무가설 : 학생들의 국어 점수의 평균은 80점이다.
대립가설 : 학생들의 국어 점수의 평균은 80점이 아니다.
"""
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv")
print(data)
print(data.describe())  #요약통계량, 국어의 mean값 : 72.900000

#정규성 검정 : one-sample t-test는 옵션. 필수는 아님.
print(stats.shapiro(data.국어)) #정규성 검정, ShapiroResult(pvalue=(0.01295975332132026)), p값이 0.05보다 작음으로 정규분포를 위배함,
#정규성 위배는 데이터 재가공이 추천됨, Wilcoxon Signed-rank test를 써야 더 안전함.
#Wilcoxon Signed-rank test는 정규성을 가정하지 않음.
from scipy.stats import wilcoxon
wilcox_result = wilcoxon(data.국어 - 80)    #평균 80점과 비교함.
print('wilcox_result : ', wilcox_result)    #pvalue=np.float64(0.39777620658898905)), 0.05보다 큼, 귀무가설 채택.
#윌콕스든 티 테스트든 둘 다 귀무가설을 채택함.
#결론 : 정규성은 부족하지만, T-test와 Wilcoxon은 같은 결과를 얻었다. 표본수가 커지면 결과는 달라질 수 있음.
#따라서 정규성 위배가 있어도, T-test 결과는 신뢰할 수 있음!

res = stats.ttest_1samp(data.국어, popmean=80)
print('statistic:%.5f, pvalue:%.5f'%res)    #statistic:-1.33218, pvalue:0.19856
#결과 : p값이 0.05보다 큼으로, 귀무가설을 채택함.
#학생들의 국어 점수의 평균은 80점이다.


"""
예제 3) 여아 신생아 몸무게 평균 검정 수행
여아 신생아의 몸무게는 평균이 2800그램으로 알려져 왔으나 이보다 더 크다는 주장이 나왔음.
표본으로 셈플데이터 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해보자

귀무가설 : 여아 신생아의 몸무게 평균은 2800그램이다.
대립가설 : 여아 신생아의 몸무게 평균은 2800그램이 아니다.(더 크거나, 더 작음)
"""
babyData = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv")
print(babyData)
print(babyData.describe())
#여아만 추출
femaleBabyData = babyData[babyData.gender == 1]
print(femaleBabyData.head(2))
print(femaleBabyData)
print(len(femaleBabyData))  #18개 맞는것을 확인
print(np.mean(femaleBabyData.weight), ' ', np.std(femaleBabyData.weight))   #3132.4444444444443   613.7878951616051
# 3132 vs 2800 : 둘 사이는 평균에 차이가 있는가? (통계적으로 검정)
# 집단이 하나일 경우, 정규성 검정은 옵션(해도 되고 안해도 됨.)
print(stats.shapiro(femaleBabyData.iloc[:, 2])) #pvalue=np.float64(0.017984789994719325)), 0.05보다 작음, 정규성 만족 못함!

#정규성 시각화(histogram으로 확인)
sns.displot(femaleBabyData.iloc[:, 2], kde=True)
plt.show()
plt.close()

#정규성 시각화(Q-Q plit으로 확인)
stats.probplot(femaleBabyData.iloc[:, 2], plot=plt)
plt.show()
plt.close()

print()

wilcox_baby = wilcoxon(femaleBabyData.weight - 2800)    #2800과 비교 방법 1 : wilcoxon
print(wilcox_baby)  #p값 : 0.03423309, 귀무 기각
resBaby = stats.ttest_1samp(femaleBabyData.weight, popmean=2800)    #2800과 비교 방법 2 : ttest_1samp
print(resBaby)      #p값 : 0.03926844, 귀무 기각

#결론 : 여아 신생아의 평균 체중은 2800그램보다 증가하였음. (대립가설 채택)