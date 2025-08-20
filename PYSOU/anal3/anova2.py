# 일원분산분석 연습

# 강남구에 있는 GS편의점 3개의 알바생 급여에 대한 평균의 차이가 있는가?
# 귀무가설 : 3개 편의점 알바생 급여에 대한 차이 없음.
# 대립가설 : 3개 편의점 알바생 급여에 대한 차이 있음.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
# data = pd.read_csv(url, header=None)
# print(data.values)
data = np.genfromtxt(url, delimiter=',')
print(data, type(data), data.shape) #<class 'numpy.ndarray'> (22, 2)

# 3개의 집단에 월급, 평균, 얻기
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]

print(gr1, ' ', np.mean(gr1))   #316.625
print(gr2, ' ', np.mean(gr2))   #256.44444444444446
print(gr3, ' ', np.mean(gr3))   #278.0

# 정규성
print(stats.shapiro(gr1).pvalue)    #0.3336828974377483, 정규성 만족
print(stats.shapiro(gr2).pvalue)    #0.6561053962402779, 정규성 만족
print(stats.shapiro(gr3).pvalue)    #0.8324811457153043, 정규성 만족

# 등분산성
print(stats.levene(gr1, gr2, gr3).pvalue)   #0.045846812634186246, 등분산성 불만족 : 일단은 만족으로 간주.
print(stats.bartlett(gr1, gr2, gr3).pvalue) #0.3508032640105389, 등분산성 만족

# 분포도 보기
plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()
plt.close()

# ANOVA 검정1 : anova_lm
df = pd.DataFrame(data, columns=['pay', 'group'])
lmodel = ols('pay ~ C(group)', data=df).fit()
print(anova_lm(lmodel, type=2)) #p : 0.043589, f : 3.711336
#결과해석 : 귀무가설 기각, 대립가설 채택. 3개 편의점 알바생 급여에 대한 차이 있음.

# ANOVA 검정2 : f_oneway
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print(f_statistic, p_value)     #p : 0.043589334959178244, f : 3.7113359882669763
#결과해석 : 귀무가설 기각, 대립가설 채택. 3개 편의점 알바생 급여에 대한 차이 있음.

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukeyResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukeyResult)
"""
group1 group2 meandiff p-adj    lower    upper  reject
------------------------------------------------------
   1.0    2.0 -60.1806 0.0355  -116.619 -3.7421   True
   1.0    3.0  -38.625 0.3215 -104.8404 27.5904  False
   2.0    3.0  21.5556 0.6802  -43.2295 86.3406  False
------------------------------------------------------
"""
tukeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()