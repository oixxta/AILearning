import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.


#tips.csv로 요약 처리 후 시각화
tips = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv')
print(tips.info())
tips['gender'] = tips['sex']
del tips['sex']                 #칼럼 이름바꾸기
print(tips.head(3))

#팁 비율 : 파생 변수 만들기!
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(3))

#그룹화 및 그룹객체 요약 통계
tip_pct_group = tips['tip_pct'].groupby([tips['gender'], tips['smoker']])
print(tip_pct_group.sum())
print(tip_pct_group.max())
print(tip_pct_group.min())

result = tip_pct_group.describe()
print(result)

#Agg기능
print(tip_pct_group.agg('sum'))
print(tip_pct_group.agg('mean'))
print(tip_pct_group.agg('var'))

#시각화 : 함수를 실행시켜서 그래프화
def myFunc(group):   #사용자 정의 함수
    diff = group.max() - group.min()
    return diff

result2 = tip_pct_group.agg(['var', 'mean', 'max', 'min', myFunc])
print(result2)
result2.plot(kind='barh', title='agg func result', stacked=True)    #누적차트
plt.show()

