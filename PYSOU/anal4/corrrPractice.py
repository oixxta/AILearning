"""
상관관계 문제)
tv,radio,newspaper 간의 상관관계를 파악하시오. 
그리고 이들의 관계를 heatmap 그래프로 표현하시오. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

#데이터 읽어오기
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv")
print(data)
print(data.describe())


#상관계수 확인
print(np.corrcoef(data.tv, data.radio)) #0.05480866
print(np.corrcoef(data.tv, data.newspaper)) #0.05664787
print(np.corrcoef(data.radio, data.newspaper))  #0.35410375

#상관관계 시각화
