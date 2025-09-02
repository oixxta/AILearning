"""
종속변수 : survived (그룹 공통)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


#EDA 실시 2 - 수치형 변수
def eda2():
    originData['age'].describe()
    print(originData['age'].skew())        #왜도
    print(originData['age'].kurt())        #첨도
    sns.histplot(originData['age'], kde=True)   #시각화 : 히스토그램
    plt.show()

    originData['bmi'].describe()
    print(originData['bmi'].skew())        #왜도
    print(originData['bmi'].kurt())        #첨도
    sns.histplot(originData['bmi'], kde=True)   #시각화 : 히스토그램
    plt.show()

    originData['cholesterol_level'].describe()
    print(originData['cholesterol_level'].skew())        #왜도
    print(originData['cholesterol_level'].kurt())        #첨도
    sns.histplot(originData['cholesterol_level'], kde=True)   #시각화 : 히스토그램
    plt.show()

    plt.close()


#EDA 실시 3 - 범주형 변수
def eda3():
    originData['gender'].value_counts()
    originData['gender'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='gender', data=originData) #시각화
    plt.title('gender')
    plt.show()

    originData['country'].value_counts()
    originData['country'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='country', data=originData) #시각화
    plt.title('country')
    plt.show()

    originData['cancer_stage'].value_counts()
    originData['cancer_stage'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='cancer_stage', data=originData) #시각화
    plt.title('cancer_stage')
    plt.show()

    originData['smoking_status'].value_counts()
    originData['smoking_status'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='smoking_status', data=originData) #시각화
    plt.title('smoking_status')
    plt.show()

    originData['hypertension'].value_counts()
    originData['hypertension'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='hypertension', data=originData) #시각화
    plt.title('hypertension')
    plt.show()

    originData['asthma'].value_counts()
    originData['asthma'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='asthma', data=originData) #시각화
    plt.title('asthma')
    plt.show()

    originData['cirrhosis'].value_counts()
    originData['cirrhosis'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='cirrhosis', data=originData) #시각화
    plt.title('cirrhosis')
    plt.show()

    originData['other_cancer'].value_counts()
    originData['other_cancer'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='other_cancer', data=originData) #시각화
    plt.title('other_cancer')
    plt.show()

    originData['treatment_type'].value_counts()
    originData['treatment_type'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='treatment_type', data=originData) #시각화
    plt.title('treatment_type')
    plt.show()

    originData['survived'].value_counts()
    originData['survived'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='survived', data=originData) #시각화
    plt.title('survived')
    plt.show()

    plt.close()

#EDA 실시 4 - 변수 간 관계 분석
def eda4():
    #수치형 vs 수치형 : bmi와 cholesterol_level 사이의 관계
    sns.scatterplot(x='bmi', y='cholesterol_level', data=originData)
    plt.title("bmi vs cholesterol_level")
    plt.show()
    print(originData[['bmi', 'cholesterol_level']].corr())
    """
                            bmi  cholesterol_level
    bmi                1.000000           0.747116
    cholesterol_level  0.747116           1.000000
    """
    #상관계수가 0.5보다 크기에, BMI와 콜레스테롤 수치 사이에는 관계가 있음.


    #범주형 vs 수치형 : survived와 cholesterol_level 사이의 관계
    sns.boxplot(x='survived', y='cholesterol_level', data=originData)
    plt.show()
    print(originData.groupby('survived')['cholesterol_level'].mean())
    """
    0    233.616645
    1    233.711997
    """
    #생존 여부와 콜레스테롤 수치는 큰 관계가 없음.


    #범주형 vs 범주형 : gender와 smoking_status 사이의 관계
    print(pd.crosstab(originData['gender'], originData['smoking_status']))
    """
    smoking_status  Current Smoker  Former Smoker  Never Smoked  Passive Smoker
    gender
    Female                  405910         406609        405985          406986
    Male                    405914         405655        405638          407303
    """
    #데이터프레임상의 남녀간의 흡연률은 거의 고르게 분포됨.


    #범주형 vs 범주형 : smoking_status와 survived 사이의 관계
    print(pd.crosstab(originData['smoking_status'], originData['survived']))
    """
    survived             0       1
    smoking_status
    Current Smoker  633346  178478
    Former Smoker   633777  178487
    Never Smoked    633810  177813
    Passive Smoker  635365  178924
    """
    #흡연 여부별 생존률은 거의 비슷한 비율로, 큰 관계가 없음.


#EDA 실시 5 - 종합 분석(전체 그림 보기)
def eda5():
    corrMatrix = originData[['bmi', 'cholesterol_level', 'hypertension', 'survived']].corr()
    sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of 4 Variables")
    plt.show()




#eda2()
#eda3()
eda4()
#eda5()
