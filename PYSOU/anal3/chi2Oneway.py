# 일원 카이제곱 검정 : 변인이 1개
# 적합도(선호도 검정)
# 실험을 통해 얻은 관찰값들이 어떤 이론적 분포를 따르고 있는지 확인하는 검정
# 꽃 색깔의 표현 분리 비율이 3:1이 맞는가?

# <적합도 검정 실습> 
"""
주사위를60회던져서나온관측도수/기대도수가아래와같이나온경우에이주사위는적합한주사위가맞는가를일원카이제곱검정
으로분석하자.

귀무가설 : 기대치와 관찰치는 차이가 없다. (현재 쓰는 주사위는 게임에 적합하다.)

대립가설 : 기대치와 관찰치는 차이가 있다. (현재 쓰는 주사위는 게임에 적합하지 않다.)
"""
import pandas as pd
import scipy.stats as stats

data = [4, 6 ,17, 16, 8, 9]     #관측값
exp = [10 ,10 ,10 ,10 ,10, 10]  #기대값
print(stats.chisquare(data))    #카이제곱값 : 14.2, p-value : 0.014
# 결론 : p-value값이 유의수준(0.05)보다 작기 때문에 귀무가설을 기각, 대립가설을 채택.
# 주사위는 게임에 적합하지 않음.
# 관측값은 우연히 발생한 것이 아닌, 어떠한 원인에 의해 발생한 필연임.

print(stats.chisquare(data, exp))   #chisquare(관찰빈도, 기대빈도(기대빈도는 생략 가능))
result = stats.chisquare(data, exp)
print('카이제곱값 : ', result[0])
print('P값 : ', result[1])

print("------------------------------")

# <선호도 검정 실습>
"""
5개의스포츠음료에대한선호도에차이가있는지검정하기

귀무가설 : 음료선호도에 차이가 없음

대립가설 : 음료선호도에 차이가 있음
"""
drinkData = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv')
print(drinkData)
print(stats.chisquare(drinkData['관측도수']))   #카이제곱값 : 20.48, p-value : 0.00039
#결론 : p-value값이 유의수준(0.05)보다 작기 때문에 귀무가설을 기각, 대립가설을 채택(선호도에 차이가 있음.).
#위 데이터를 시각화 하기 : 어떤 음료가 기대보다 많이 선택되었는지 확인
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.
total = drinkData['관측도수'].sum()
expected = [total / len(drinkData)] * len(drinkData)
print('expected : ', expected)

x = np.arange(len(drinkData))
width = 0.35    #막대그래프에 쓸 너비
plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, drinkData['관측도수'], width=width, label='관측도수')
plt.bar(x - width / 2, expected, width=width, label='기대도수', alpha=0.6)
plt.xticks(x, drinkData['음료종류'])
plt.xlabel('음료 종류')
plt.ylabel('도수')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# 그래프와 카이제곱 검정 결과를 바탕으로 어떤 음료가 더 인기있는지 구체적으로 분석
# 총합과 기대도수는 이미 구함.
# 차이 계산
drinkData['기대도수'] = expected
drinkData['차이(관측-기대)'] = drinkData['관측도수'] - drinkData['기대도수']
drinkData['차이비율(%)'] = round(drinkData['차이(관측-기대)'] / expected * 100, 2)
drinkData.sort_values(by='차이(관측-기대)', ascending=False, inplace=True)
drinkData.reset_index(drop=True, inplace=True)
print(drinkData)