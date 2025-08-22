#공분산 & 상관계수 확인 연습
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')


data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv")
#print(data.head(3))
print(data.describe())
print(np.std(data.친밀도))  #0.970345
print(np.std(data.적절성))  #0.859657
print(np.std(data.만족도))  #0.828744

#plt.hist([np.std(data.친밀도), np.std(data.적절성), np.std(data.만족도)])
#plt.show()
#plt.close()

print('공분산-----')
print(np.cov(data.친밀도, data.적절성)) #numpy는 한번에 두 개씩만 공분산을 확인할 수 있음.
print(np.cov(data.친밀도, data.만족도))
print(data.cov())             #dataframe으로 공분산 출력, 3개 이상의 것들을 확인할 때 더 효율적

print('상관계수-----')
print(np.corrcoef(data.친밀도, data.적절성)) #numpy는 한번에 두 개씩만 상관계수를 확인할 수 있음.
print(np.corrcoef(data.친밀도, data.만족도))
print(data.corr())             #dataframe으로 상관계수 출력, 3개 이상의 것들을 확인할 때 더 효율적
print(data.corr(method='pearson'))  #피어슨 상관계수, 변수가 등간, 비율척도일때. 기본값.
print(data.corr(method='spearman')) #스피어만 상관계수, 변수가 서열척도일때.
print(data.corr(method='kendall'))  #켄달 상관계수, 스피어만과 비슷함.

# 예) 만족도에 대한 다른 특성(변수) 사이의 상관관계 보기.
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))
#만족도    1.000000
#적절성    0.766853
#친밀도    0.467145

# 상관관계 시각화
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()
plt.close()

from pandas.plotting import scatter_matrix
attr = ['친밀도', '적절성', '만족도']
scatter_matrix(data[attr], figsize=(10, 6))
plt.show()
plt.close()

import seaborn as sns
sns.heatmap(data.corr())
plt.show()
plt.close()