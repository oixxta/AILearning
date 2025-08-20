# 이원 카이제곱
# 동질성 : 
# 검정 두 집단의 분포가 동일한가 다른 분포인가를 검증하는 방법. 두 집단 이상에서 각 범주집단 간의 분포가
# 동일한가를 검정하게 됨. 두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다


# 동질성 검정실습 1 교육방법에 따른 교육생들의 만족도 분석 동질성 검정

import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv")
print(data.head(3))
print(data['method'].unique())  #[1 2 3]
print(set(data['survey']))      #{1, 2, 3, 4, 5}

#교차표
ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)

#동질성 검증(방법은 일원 카이제곱의 동립성 검증과 비슷함.)
chi2, p, ddof, _ = stats.chi2_contingency(ctab)
print('test statistic : ', chi2, ', p : ', p, ', ddof : ', ddof)
#test statistic :  6.544667820529891 , p :  0.5864574374550608 , ddof :  8

#결론 : 유의수준 0.05보다 p값이 크기 때문에, 귀무가설을 채택함.(우연히 발생된 데이터로 간주함)
#교육방법에 따른 교육생들의 만족도는 아무 관련 없음.



# 동질성 검정실습2) 연령대별sns이용률의동질성검정
#20대에서40대까지연령대별로서로조금씩그특성이다른SNS 서비스들에대해이용현황을조사한자료를바탕으로연령대별로홍보
#전략을세우고자한다.
#연령대별로이용현황이서로동일한지검정해보도록하자

# 귀무가설 : 연령대별 SNS 서비스별 이용 현황은 동일하다
# 대립가설 : 연령대별 SNS 서비스별 이용 현황은 동일하지 않다.

data2 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv")
print(data2)
print(data2['age'].unique())        #[1 2 3]
print(data2['service'].unique())    #['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service'], margins=True)
print(ctab2)
chi2, p, ddof, _ = stats.chi2_contingency(ctab2)
print('test statistic : ', chi2, ', p : ', p, ', ddof : ', ddof)
#test statistic :  102.75202494484225 , p :  3.916549763057839e-15 , ddof :  15
#결론 : 유의수준 0.05보다 p값이 작기 때문에, 귀무가설을 기각함.(두 데이터는 동질성을 가짐)

#위 데이터는 사실 셈플데이터로, 샘플링 연습을 위해 위 데이터를 모집단이라 가정하자.
#그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본들 추출해 처리해 보자.
sampleData = data2.sample(n = 50, replace = True, random_state=1)   #random_state : 시드넘버 고정
print(sampleData)

ctab3 = pd.crosstab(index=sampleData['age'], columns=sampleData['service'], margins=True)
print(ctab3)
chi2, p, ddof, _ = stats.chi2_contingency(ctab3)
print('test statistic : ', chi2, ', p : ', p, ', ddof : ', ddof)
#test statistic :  8.465143862871136 , p :  0.9037848671247912 , ddof :  15

