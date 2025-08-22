#회귀분석 연습문제
"""
회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용

나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  
"""
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#데이터 가져오기
data = pd.read_csv("tvData.csv")
data.drop(9, inplace=True) #이상치 제거 : 9번째
jisangpa = data.지상파  #지상파 결측치 1개를 지상파 평균값으로 대체.
jisangpa.fillna(jisangpa.mean(), inplace=True)
jongpyeoon = data.종편
undong = data.운동



print(undong)
def lm1():
    #상관관계 확인하기 : 지상파-운동
    print(np.corrcoef(jisangpa, undong))    #-0.87418419, 두개는 강한 상관관계를 가짐. 그래프 우하향, 회귀분석 가능.
    #plt.scatter(jisangpa, undong)
    plt.show()
    plt.close()

    #회귀분석 하기 : 지상파-운동, scipy.stats.linregress()
    model1 = stats.linregress(x=jisangpa, y=undong)
    print('slope : ', model1.slope)         # -0.6754562043795621
    print('intercept : ', model1.intercept) # 4.719815412689501
    #위 회귀분석으로 만든 수식 : y = -0.6754562043795621 * x + 4.719815412689501
    #plt.scatter(jisangpa, undong)
    #plt.plot(jisangpa, model1.slope * jisangpa + model1.intercept)
    plt.show()
    plt.close()
    #수식 검증 : 지상파-운동
    print('지상파 시청 시간이 0.9일때의 운동시간 예측 : ', model1.slope * 0.9 + model1.intercept)   #4.111904828747895, 실제값 4.2와 거의 근접
    #운동시간 예측 : 지상파-운동
    print('지상파 시청 시간이 9일때의 운동시간 예측 : ', model1.slope * 9 + model1.intercept)   #-1.3592904267265578


    #회귀분석 하기 : 지상파-운동, statsmodels ols()
    import statsmodels.formula.api as smf
    df2 = pd.concat([jisangpa, undong], axis=1)
    df2.columns = ('x1', 'x2')
    model2 = smf.ols(formula='x2 ~ x1', data=df2).fit()
    print(model2.summary())
    """
                                    OLS Regression Results
    ==============================================================================
    Dep. Variable:                     x2   R-squared:                       0.764
    Model:                            OLS   Adj. R-squared:                  0.745
    Method:                 Least Squares   F-statistic:                     38.89
    Date:                Fri, 22 Aug 2025   Prob (F-statistic):           4.34e-05
    Time:                        15:13:59   Log-Likelihood:                -10.281
    No. Observations:                  14   AIC:                             24.56
    Df Residuals:                      12   BIC:                             25.84
    Df Model:                           1
    Covariance Type:            nonrobust
    ==============================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      4.7198      0.312     15.135      0.000       4.040       5.399
    x1            -0.6755      0.108     -6.236      0.000      -0.911      -0.439
    ==============================================================================
    Omnibus:                        0.112   Durbin-Watson:                   2.588
    Prob(Omnibus):                  0.945   Jarque-Bera (JB):                0.091
    Skew:                          -0.088   Prob(JB):                        0.956
    Kurtosis:                       2.646   Cond. No.                         6.76
    ==============================================================================
    절편(Intercept) : 4.7198, 독립변수 x1의 기울기 : -0.6755.
    위 결과로 만든 수식 : y = -0.6755 * x + 4.7198
    """

    #회귀분석 하기 : 지상파-운동, LinearRegression
    from sklearn.linear_model import LinearRegression
    model3 = LinearRegression()
    #jisanpaArr = jisangpa.values
    #undongArr = undong.values
    #fitModel = model3.fit(jisanpaArr, undongArr)
    #print(fitModel.coef_)
    #print(fitModel.intercept_)



def lm2():
    #상관관계 확인하기 : 지상파-종편
    print(np.corrcoef(jisangpa, jongpyeoon))    #0.87299212, 두개는 강한 상관관계를 가짐. 그래프 우상향, 회귀분석 가능.
    plt.scatter(jisangpa,jongpyeoon)
    plt.show()
    plt.close()

    #회귀분석 하기 : 지상파-종편
    model2 = stats.linregress(x=jisangpa, y=jongpyeoon)
    print('slope : ', model2.slope)         # 0.765602189781022
    print('intercept : ', model2.intercept) # 0.32208761129381536
    #위 회귀분석으로 만든 수식 : y = 0.765602189781022 * x + 0.32208761129381536
    plt.scatter(jisangpa, jongpyeoon)
    plt.plot(jisangpa, model2.slope * jisangpa + model2.intercept)
    plt.show()
    plt.close()

    #수식 검증 : 지상파-종편
    print('지상파 시청 시간이 0.9일때의 종편시청시간 예측 : ', model2.slope * 0.9 + model2.intercept)   #1.0111295820967352, 실제값 0.7과 거의 근접

    #종편 시청시간 예측 : 지상파-종편
    print('지상파 시청 시간이 9일때의 종편시청시간 예측 : ', model2.slope * 9 + model2.intercept)   #7.212507319323014


lm1()
#lm2()