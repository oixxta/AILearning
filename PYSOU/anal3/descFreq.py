#기술통계의 목적은 데이터를 수지, 요약, 정리, 시각화 하는 것임.
#도수분포표(Frequancy Distribution Table)는 데이터를 구간별로 나눠 빈도를 정리한 표다.
#이를 통해 데이터의 분포를 한 눈에 파악할 수 있다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.

# step1 : 데이터를 읽어서 데이터 프레임에 저장
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')
#print(df.head(3))  #키 칼럼만 있는 키에 대한 데이터

# step2 : 최대값과 최소값 구하기
min_height = df['키'].min()     #최소값
max_height = df['키'].max()     #최대값
print(min_height, max_height)

# step3 : 구간설정 하기 (cut 사용)
bins = np.arange(156, 195, 5) #156부터 195사이의, 5개의 구간
print(bins)
df['계급'] = pd.cut(df['키'], bins=bins, right=True, include_lowest=True)
print(df.head(3))
print(df.tail(3))

# step4 : 각 구간의 중앙값 구하기 (156 + 161) / 2
df['계급값'] = df['계급'].apply(lambda x:int((x.left + x.right) / 2))
print(df.head(3))
print(df.tail(3))

# step5 : 도수분표표를 만들기 위한 도수를 계산
freq = df['계급'].value_counts().sort_index()

# step6 : 상대 도수(전체 데이터에 대한 비율) 계산
relatibe_freq = (freq / freq.sum()).round(2)
print(relatibe_freq)

# step7 : 누적 도수 계산 - 구간별 도수를 누적하기
cum_freq = freq.cumsum()
print(cum_freq)

# step8 : 도수 분포표 작성
dist_table = pd.DataFrame({
    # "156 ~ 161" 이런 모양 출력하기 [(156, 161)]
    '계급' : [f"{int(interval.left)} ~ {int(interval.right)}" for interval in freq.index],
    # 계급의 중간값
    '계급값' : [int(((interval.left) + (interval.right)) / 2) for interval in freq.index],
    '도수' : freq.values,
    '상대도수' : relatibe_freq.values,
    '누적도수' : cum_freq.values
})
print('도수 분포표')
print(dist_table)

"""
     계급        계급값    도수  상대도수  누적도수
0    155 ~ 161    158      5     0.10        5
1    161 ~ 166    163      8     0.16       13
2    166 ~ 171    168     10     0.20       23
3    171 ~ 176    173     13     0.26       36
4    176 ~ 181    178      6     0.12       42
5    181 ~ 186    183      5     0.10       47
6    186 ~ 191    188      3     0.06       50
"""

# step9 : 히스토그램 그리기
plt.figure(figsize=(8,5))
plt.bar(dist_table['계급값'], dist_table['도수'], width=5, color='cornflowerblue', edgecolor='black')         #세로막대그래프
plt.title('학생 50명 키 히스토그램', fontsize=16)
plt.xlabel('키(계급값)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
