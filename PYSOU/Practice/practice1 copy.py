"""
생활습관 및 건강지표(bmi, smoking_status, colesterol_level, hypertension, asthma, cirrhosis, other_cancer)와 사망률이 유의미한 관계가 있다.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import levene
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.font_manager as fm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def print_stats(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우 기준
fontprop = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=fontprop)
plt.rcParams['axes.unicode_minus'] = False

#데이터 긁어오기
originData = pd.read_csv("lung_cancer_mortality_data_large_v2.csv")        # 원본 흡연 데이터

#EDA 실시 1 - 데이터 기본 정보 확인
print('데이터의 상위 5개 : \n', originData.head(5))
"""
    id   age  gender         country diagnosis_date cancer_stage beginning_of_treatment_date  ... hypertension asthma  cirrhosis  other_cancer  treatment_type  end_treatment_date  survived
0   1  64.0  Female         Germany     2016-04-07    Stage III                  2016-04-21  ...            1      1          0             0        Combined          2017-11-15         0
1   2  50.0    Male  Czech Republic     2023-04-22    Stage III                  2023-05-02  ...            1      0          0             0       Radiation          2024-04-25         0
2   3  65.0    Male         Romania     2023-04-07     Stage IV                  2023-04-12  ...            0      0          0             0         Surgery          2025-03-11         0
3   4  51.0  Female          Latvia     2016-02-07    Stage III                  2016-02-13  ...            1      1          1             0         Surgery          2017-04-14         1
4   5  37.0    Male          Greece     2023-12-01      Stage I                  2023-12-03  ...            0      0          0             0    Chemotherapy          2024-09-20         0
"""
print('데이터의 행/열 : \n', originData.shape)              #(3250000, 18)
print('데이터의 타입 및 결측치: \n', originData.info())
"""
 #   Column                       Dtype
---  ------                       -----
 0   id                           int64                 아이디
 1   age                          float64   수치형       나이
 2   gender                       object    범주형       성별
 3   country                      object    범주형       국적
 4   diagnosis_date               object    수치형       진단일
 5   cancer_stage                 object    범주형       암 진행상황
 6   beginning_of_treatment_date  object    수치형       치료 시작일
 7   family_history               object    범주형       가족력 여부
 8   smoking_status               object    범주형       흡연상태
 9   bmi                          float64   수치형       BMI
 10  cholesterol_level            int64     수치형       콜레스테롤 레벨
 11  hypertension                 int64     범주형       고혈압
 12  asthma                       int64     범주형       천식
 13  cirrhosis                    int64     범주형       간경병증
 14  other_cancer                 int64     범주형       다른 암
 15  treatment_type               object    범주형       치료 방법
 16  end_treatment_date           object    수치형       치료 종료일
 17  survived                     int64     범주형       생존여부
"""
print('데이터 칼럼 : \n', originData.columns)
#['id', 'age', 'gender', 'country', 'diagnosis_date', 'cancer_stage', 'beginning_of_treatment_date', 'family_history', 'smoking_status','bmi', 'cholesterol_level', 'hypertension', 'asthma', 'cirrhosis','other_cancer', 'treatment_type', 'end_treatment_date', 'survived']
print('데이터 결측치 : \n', originData.isnull().sum())      # 0

#데이터 전처리 실시
#전처리 1 : age, bmi, cholesterol_level 중 상식적 + 현실에서 비정상적인 값들 드랍
originData = originData[(originData['age'] >= 20) & (originData['age'] <= 80)]
originData = originData[(originData['bmi'] >= 10) & (originData['bmi'] <= 50)]
originData = originData[(originData['cholesterol_level'] >= 100) & (originData['cholesterol_level'] <= 300)]
print(originData.shape)     #(3231831, 18), 2만5천개 정도 드랍

#전처리 2 : 새 칼럼(DangerScore)를 추가하고, 특정 범주형 변수의 값이 1일때마다 1씩 추가되는 점수로 채워넣기
danger_cols = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer']
originData['DangerScore'] = originData[danger_cols].sum(axis=1)

#전처리 3 : 새 칼럼(CureDay)을 추가하고 end_treatment_date에서 beginning_of_treatment_date뺀 값 넣기
# 날짜형 변환
originData['beginning_of_treatment_date'] = pd.to_datetime(originData['beginning_of_treatment_date'])
originData['end_treatment_date'] = pd.to_datetime(originData['end_treatment_date'])

# 차이 계산 (일 단위)
originData['CureDay'] = (originData['end_treatment_date'] - originData['beginning_of_treatment_date']).dt.days

#전처리 4 : 수치형 데이터를 필요에 따라 범주형으로 정리(예: BMI : 저체중, 정상, 과체중, 비만, 고도 비만)
def bmi_category(bmi):
    if bmi < 18.5:
        return '0'  #'저체중'
    elif bmi < 25:
        return '1'  #'정상'
    elif bmi < 30:
        return '2'  #'과체중'
    elif bmi < 35:
        return '3'  #'비만'
    else:
        return '4'  #'고도 비만'

originData['bmi_category'] = originData['bmi'].apply(bmi_category)

#전처리 결과 확인 : 
print(originData.head(5))

# 가설1 : bmi와 survived는 관계가 있다.
#사용 검정함수 : t-test
#귀무가설 H₀: 생존 여부에 따라 BMI의 평균은 차이가 없다.
#대립가설 H₁: 생존 여부에 따라 BMI의 평균은 차이가 있다.
#신뢰수준 알파값 95%로 설정. (유의수준 0.05)
#선행조건 : 두 집단의 자료는 정규분포를 따르며, 분산이 동일하다(등분산성)
#결과 : 생존 여부에 따라 BMI 평균은 통계적으로 유의미한 차이가 없다. (귀무가설 채택)
def question1():
    bmi_survived = originData[originData['survived'] == 1]['bmi']
    bmi_died = originData[originData['survived'] == 0]['bmi']
    t_stat, p_value = stats.ttest_ind(bmi_survived, bmi_died, equal_var=False)
    print('Q1 p_value : ', p_value)
    alpha = 0.05
    if p_value < alpha: # 생존 여부에 따라 BMI 평균은 통계적으로 유의미한 차이가 없다. (귀무가설 채택)
        print("=> 생존 여부에 따라 BMI 평균은 통계적으로 유의미한 차이가 있다. (귀무가설 기각)")
    else:
        print("=> 생존 여부에 따라 BMI 평균은 통계적으로 유의미한 차이가 없다. (귀무가설 채택)")

    #등분산성 검정
    levene_stat, levene_p = levene(bmi_survived, bmi_died)
    if(levene_p > 0.05):
        print("분산이 같다고 할 수 있다. 등분산성이다.")
    else:
        print("분산이 같다고 할 수 없다. 등분산성 가정이 부적절하다")
    #등분산성 부적절, 따라서 Welch’s t-test도 돌려봄.
    welch_result = stats.ttest_ind(bmi_survived, bmi_died, equal_var=False)
    print(welch_result) #pvalue=np.float64(0.7865640462411191)

    #시각화
    sns.boxplot(x='survived', y='bmi', data=originData)
    plt.title('BMI distribution by Survival')
    plt.xlabel('survived')
    plt.ylabel('bmi')
    plt.legend()
    plt.show()
    plt.close()

# 가설2 : asthma와 survived는 관계가 있다.
#사용 검정함수 : 카이제곱 검정
#귀무가설 (H₀): 천식 유무는 생존 여부에 영향을 미치지 않는다.
#대립가설 (H₁): 천식 유무는 생존 여부에 영향을 미친다.
#신뢰수준 알파값 95%로 설정. (유의수준 0.05)
#선행조건 : 두 집단의 자료는 정규분포를 따르며, 분산이 동일하다(등분산성)
#결과 : 천식 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)
def question2():
    crossTable = pd.crosstab(originData['asthma'], originData['survived'])
    print(crossTable)
    # survived        0       1
    # asthma
    # 0         1347419  379215
    # 1         1188879  334487
    chi2, p_value, dof, expected = chi2_contingency(crossTable)
    print('카이제곱 : ', chi2)
    print('p value : ', p_value)
    alpha = 0.05
    if p_value < alpha: # 천식 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)
        print("=> 천식 유무는 생존 여부에 영향을 미친다. (귀무가설 기각)")
    else:
        print("=> 천식 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)")
    
    y_true = originData['survived']
    y_pred = originData['asthma']  # 가설에서 asthma가 survived를 예측한다고 간주
    print("\n[모델 성능 지표]")
    print_stats(y_true, y_pred)

    #시각화
    sns.countplot(x='asthma', hue='survived', data=originData)
    plt.xlabel('천식 유무 (0: 없음, 1: 있음)')
    plt.ylabel('환자 수')
    plt.legend(title='생존 여부', labels=['사망 (0)', '생존 (1)'])
    plt.title('천식 여부별 생존자 수 분포')
    plt.show()
    plt.close()

# 가설3 : cirrhosis와 survived는 관계가 있다.
#사용 검정함수 : 카이제곱 검정
#귀무가설 (H₀): 간경변 유무는 생존 여부에 영향을 미치지 않는다
#대립가설 (H₁): 간경변 유무는 생존 여부에 영향을 준다.
#신뢰수준 알파값 95%로 설정. (유의수준 0.05)
#결과 : 간경변 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)
def question3():
    crossTable = pd.crosstab(originData['cirrhosis'], originData['survived'])
    print(crossTable)
    # survived         0       1
    # cirrhosis
    # 0          1961493  552112
    # 1           574805  161590
    chi2, p_value, dof, expected = chi2_contingency(crossTable)
    alpha = 0.05
    print('카이제곱 : ', chi2)
    print('p value : ', p_value)
    if p_value < alpha: # 간경변 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)
        print("=> 간경변 유무는 생존 여부에 영향을 준다. (귀무가설 기각)")
    else:
        print("=> 간경변 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)")
    print("\n[모델 성능 지표]")
    y_true = originData['survived']
    y_pred = originData['cirrhosis']
    print_stats(y_true, y_pred)


    #시각화
    sns.countplot(x='cirrhosis', hue='survived', data=originData)
    plt.xlabel('간경변 유무 (0: 없음, 1: 있음)')
    plt.ylabel('환자 수')
    plt.legend(title='생존 여부', labels=['사망 (0)', '생존 (1)'])
    plt.title('간경변 여부별 생존자 수 분포')
    plt.show()
    plt.close()


# 가설4 : other_cancer와 suvrived는 관계가 있다.
#사용 검정함수 : 카이제곱 검정
#귀무가설 (H₀): 다른 암 보유 유무는 생존 여부에 영향을 미치지 않는다
#대립가설 (H₁): 다른 암 보유 유무는 생존 여부에 영향을 준다.
#신뢰수준 알파값 95%로 설정. (유의수준 0.05)
#결과 : 다른 암 보유 유무는 생존 여부에 영향을 준다. (귀무가설 기각)
def question4():
    crossTable = pd.crosstab(originData['other_cancer'], originData['survived'])
    print(crossTable)
    #survived            0       1
    #other_cancer
    #0             2312551  651360
    #1              223747   62342
    chi2, p_value, dof, expected = chi2_contingency(crossTable)
    alpha = 0.05
    print('카이제곱 : ', chi2)
    print('p value : ', p_value)
    if p_value < alpha: # 다른 암 보유 유무는 생존 여부에 영향을 준다. (귀무가설 기각)
        print("=> 다른 암 보유 유무는 생존 여부에 영향을 준다. (귀무가설 기각)")
    else:
        print("=> 다른 암 보유 유무는 생존 여부에 영향을 미치지 않는다. (귀무가설 채택)")
    y_true = originData['survived']
    y_pred = originData['cirrhosis']
    print_stats(y_true, y_pred)
    
    #시각화
    sns.countplot(x='other_cancer', hue='survived', data=originData)
    plt.xlabel('다른 암 보유 유무 (0: 없음, 1: 있음)')
    plt.ylabel('환자 수')
    plt.legend(title='생존 여부', labels=['사망 (0)', '생존 (1)'])
    plt.title('다른 암 보유 여부별 생존자 수 분포')
    plt.show()
    plt.close()


#eda2()
#eda3()
#eda4()
#eda5()
#question1()
#question2()
#question3()
question4()