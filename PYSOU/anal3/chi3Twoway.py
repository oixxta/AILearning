# 이원 카이제곱 검정 : 교차분할표를 사용, 변인이 2개
# 변인이 두 개 - 독립성 또는 동질성


# <독립성(관련성) 검정 실습>
"""
교육수준과흡연율간의관련성분석

귀무가설 : 교육수준과 흡연율 간의 관련이 없다(독립이다, 연관성이 없다)

대립가설 : 교육수준과 흡연율 간의 관련이 있다(독립이 아니다, 연관성이 있다)
"""
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv')
#print(data)
print(data['education'].unique())   #[1 2 3], 의미부여 필요
print(data['smoking'].unique())     #[1 2 3], 의미부여 필요

# 학력별 흡연 인원수를 위한 교차표
ctab = pd.crosstab(index=data['education'], columns=data['smoking'])
#ctab = pd.crosstab(index=data['education'], columns=data['smoking'], normalize=True)    #빈도수(비율)로 나옴
print()
ctab.index = ["대학원졸", "대졸", "고졸"]
ctab.columns = ["과흡연", "보통", "비흡연"]
print(ctab)
chi2, p, dof, _ = stats.chi2_contingency(ctab)
msg = ("Test statics : {}, P-value : {}, df : {}")
print(msg.format(chi2, p, dof)) #카이제곱값 : 18.91, P-value : 0.00081, 자유도 : 4, p값과 카이제곱값은 반비례 관계임.
#결론 : p값이 유의수준 0.05보다 작기 때문에, 귀무가설 기각, 대립가설 채택. 교육수준과 흡연율은 관련이 있다.

print()

# <독립성(관련성) 검정 실습>
"""
음료 종류와 성별 간의 선호도 차이 검정

귀무가설 : 셩별과 음료 선호는 서로 관련이 없다

대립가설 : 성별과 음료 선호는 서로 관련이 있다(성별에 따라 음료의 선호가 다르다)
"""
data = pd.DataFrame({
    'getoray' : [30, 20],
    'pocary' : [20, 30],
    'vita500' : [10 , 30],
}, index=['male', 'female'])
print(data)
#이원 카이제곱 검정
chi2, p, dof, expected = stats.chi2_contingency(data)
print("카이제곱 통계량  : ", chi2)  #11.375
print("p값(유의확률) : ", p)       #0.0033
print("자유도  : ", dof)           #2
print("기대도수 : ", expected)     #[21.42857143 21.42857143 17.14285714]
#결론 : p값이 0.05보다 작기 때문에, 귀무가설 기각, 대립가설 채택. 성별에 따라 음료의 선호가 다르다

# 시각화 : heatmap
# 히트맵은 색상을 활용해 값의 분포를 보여주는 방법.
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.

sns.heatmap(data, annot=True, fmt='d', cmap='Blues')   #annot : 숫자값 표시 여부
plt.title('성별에 따른 음료 선호 (빈도)')
plt.xlabel('음료')
plt.ylabel('성별')
plt.show()