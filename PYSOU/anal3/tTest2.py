"""
독립표본 검정 (Independent sample t-test) : 두 집단의 평균의 차이 검정
서로 다른(독립인) 두 집단의 평균에 대한 통계 검정에 주로 사용된다.
비교를 위해 평균과 표준편차 통계량을 사용한다.
평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확인해
분석 대상인 두 자료가 같을 가능성이 우연의 범위 5%에 들어가는지를
판별한다.

결국, t-test는 두 집단의 평균과 표준편차 비율에 대한 대조 검정법이다.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from scipy.stats import levene

"""
실습1) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정.

귀무가설 : 두 집단간 파이썬 시험 점수 평균의 차이는 없다.
대립가설 : 두 집단간 파이썬 시험 점수 평균의 차이가 있다.
신뢰구간 95%로 설정. (유의수준 0.05)
선행조건 : 두 집단의 자료는 정규분포를 따르며, 분산이 동일하다(등분산성)
"""
def practice1():
    male = [75, 85, 100, 72.5, 86.5]
    female = [63.2, 76, 52, 100, 70]
    print(np.mean(male), ' ', np.mean(female))  #남자평균 : 83.8   여자평균 : 72.24

    two_sample = stats.ttest_ind(male, female)  #독립표본 Ttest
    print(two_sample)   #pvalue=np.float64(0.2525076844853278)

    #결론 : p값(0.2525)은 유의수준 0.05보다 크기 때문에, 귀무가설을 채택,
    #두 집단간 파이썬 시험 점수 평균의 차이는 없다.

    two_sample = stats.ttest_ind(male, female, equal_var=True)  #equal_var=True를 줄 경우, 두 집단의 분산이 동일하다고 가정함, 기본값.
    print(two_sample)   #pvalue=np.float64(0.2525076844853278)

    #등분산 검정(두 집단의 분산이 같은지 검정) : levene 방식으로
    levene_stat, levene_p = levene(male, female)
    print(f"통계량 : {levene_stat:.4f}, p-value : {levene_p:.4f}")
    if(levene_p > 0.05):
        print("분산이 같다고 할 수 있다. 등분산성이다.")
    else:
        print("분산이 같다고 할 수 없다. 등분산성 가정이 부적절하다")
    
    #만약, 등분산성 가정이 부적절한 경우, Welch’s t-test 사용이 권장됨.
    welch_result = stats.ttest_ind(male, female, equal_var=False)
    print(welch_result)


practice1()


