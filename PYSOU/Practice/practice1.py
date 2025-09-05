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


# 준비된 데이터를 갖고 확인해볼 만한 가설 세워보기(3개)
#가설1 : 최고위험군(DangerScore 값이 4)일수록, 폐암 사망률이 높다.
#가설2 : 폐암 투병기간과 사망률은 관련이 있다
#가설3 : 고도비만(bmi 값이 4)일수록, 폐암 사망률이 높다.


# 3개 중 가장 그럴사한 것 검정해보기 : 귀무/대립 가설 세우고 근거 제시
#선택 가설 : 3번, 고도비만(bmi 값이 3 또는 4)일수록, 폐암 사망률이 높다.
#귀무가설 : BMI 값과 생존률은 아무 상관 없음.
#대립가설 : BMI 값과 생존률은 상관 있음.(BMI 수치가 높을수록, 사망률이 더 높다.)
#근거 : 뚱뚱할수록 운동을 안해서 면역력이 떨어질 것이기 때문에 질병에 더 취약함.


# 가설에 따라 적절한 검정 함수 선택하기
#사용 검정함수 : 카이제곱 독립성 검정
#연속형 데이터 칼럼 'bmi'를 수치에 따라 0 ~ 5 사이의 값을 갖는 범주형 데이터로 바꿈. 그리고,
#전체 데이터를 그룹1(bmi 값이 3,4), 그룹2(bmi 값이 1,2)로 나누고, 양쪽의 사망률(survived값이 1인 비율) 비교함.


# 탐색적 데이터 분석(EDA) : 시각화
def edaBMI():
    originData['bmi_category'].value_counts()
    originData['bmi_category'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='bmi_category', data=originData) #시각화
    plt.legend()
    plt.title('bmi_category')
    plt.show()
    
def edaSurvived():
    originData['survived'].value_counts()
    originData['survived'].value_counts(normalize=True)  # 비율로 보기
    sns.countplot(x='survived', data=originData) #시각화
    plt.legend()
    plt.title('survived')
    plt.show()

def edaBMIxSurvived():
    grouped_bmi = originData[originData['bmi_category'].isin(['1', '2', '3', '4'])].copy()
    grouped_bmi['bmi_group'] = grouped_bmi['bmi_category'].apply(lambda x: 'high' if x in ['3', '4'] else 'normal')
    sns.countplot(x='bmi_group', hue='survived', data=grouped_bmi)
    plt.title('BMI distribution by Survival')
    plt.xlabel('bmi')
    plt.ylabel('survived')
    plt.legend()
    plt.show()
    plt.close()

#edaBMI()
#edaSurvived()
#edaBMIxSurvived()


# 가설검정 진행
# 고도비만 그룹(3,4) vs 정상/과체중 그룹(1,2)
grouped_bmi = originData[originData['bmi_category'].isin(['1', '2', '3', '4'])].copy()
grouped_bmi['bmi_group'] = grouped_bmi['bmi_category'].apply(lambda x: 'high' if x in ['3', '4'] else 'normal')
# 교차표 만들기
crossTable = pd.crosstab(grouped_bmi['bmi_group'], grouped_bmi['survived'])
print("교차표:\n", crossTable)
# 카이제곱 검정
chi2, p, dof, expected = chi2_contingency(crossTable)
print(f"\n카이제곱 통계량 = {chi2:.3f}, p-value = {p:.5f}")
# 유의수준
alpha = 0.05
if p < alpha:
    print("=> 귀무가설 기각: BMI 그룹과 생존여부는 관련이 있음.")
else:
    print("=> 귀무가설 채택: BMI 그룹과 생존여부는 관련이 없음.")
#카이제곱 통계량 = 0.047, p-value = 0.82796
#=> 귀무가설 채택: BMI 그룹과 생존여부는 관련이 없음.


# 머신러닝 모델링 : 특성(feature) 준비
#- 결측치 처리
originData.dropna(inplace=True)
#- 모델링용 특성 선택
data = originData[['bmi_category', 'survived']].copy()
feature = data.drop('survived', axis=1)
label = data['survived']

# 머신러닝 모델링 : 훈련 / 테스트 데이터 분할
xTrain, xTest, yTrain, yTest = train_test_split(feature, label, test_size=0.2, random_state=1)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(2585464, 1) (646367, 1) (2585464,) (646367,)

# 머신러닝 모델링 : 모델 훈련(RandomForestClassifier)
rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
rfModel.fit(xTrain, yTrain)
pred = rfModel.predict(xTest)
print("Confusion Matrix:\n", confusion_matrix(yTest, pred))
# Confusion Matrix:
# [[504549      0]
# [141818      0]]
print("\nClassification Report:\n", classification_report(yTest, pred))
#Classification Report:
#               precision    recall  f1-score   support
#
#           0       0.78      1.00      0.88    504549
#           1       0.00      0.00      0.00    141818
#
#    accuracy                           0.78    646367
#   macro avg       0.39      0.50      0.44    646367
#weighted avg       0.61      0.78      0.68    646367


# 머신러닝 모델링 : 특성 중요도 분석
importances = pd.Series(rfModel.feature_importances_, index=feature.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("특성 중요도")
plt.tight_layout()
plt.show()


# 시각화 : 위험도 매트릭스
# bmi_category별 사망률 계산
risk_matrix = originData.groupby('bmi_category')['survived'].apply(lambda x: (x == 0).mean())
# 인덱스 숫자 정렬
risk_matrix.index = risk_matrix.index.astype(int)
risk_matrix = risk_matrix.sort_index()
# 시각화 (1행 n열로 보기 위해 .to_frame().T 사용)
plt.figure(figsize=(8, 2))
sns.heatmap(risk_matrix.to_frame().T, annot=True, cmap='Reds', cbar=True, fmt=".2f")
plt.title("BMI별 사망률 위험도 매트릭스")
plt.xlabel("범주형 BMI")
plt.yticks([])  # y축 레이블 제거
plt.tight_layout()
plt.show()


#최종 결론
#통계적으로 유의미하지 않음. BMI와 사망률은 무관함.
#해당 결론은 이미 가설검정 단계에서 명확하게 확인되었으며, 머신러닝 모델에서도 1개의 feature로는 예측할 수 없었고,
#위험도 매트릭스로 확인한 것으로도 BMI 수치와 폐암으로 인한 사망률 차이는 통계적으로 큰 차이가 없었음.