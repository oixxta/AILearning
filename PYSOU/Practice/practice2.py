import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# 한글 폰트 설정 (한글 깨짐 방지)
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 사용 시
# matplotlib.rcParams['font.family'] = 'AppleGothic'  # Mac 사용 시
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
originData = pd.read_csv("lung_cancer_mortality_data_large_v2.csv")

# 전처리: 나이, BMI, 콜레스테롤 정상 범위 내로 필터링
originData = originData[(originData['age'] >= 20) & (originData['age'] <= 80)]
originData = originData[(originData['bmi'] >= 10) & (originData['bmi'] <= 50)]
originData = originData[(originData['cholesterol_level'] >= 100) & (originData['cholesterol_level'] <= 300)]

# 날짜 전처리
originData['beginning_of_treatment_date'] = pd.to_datetime(originData['beginning_of_treatment_date'])
originData['end_treatment_date'] = pd.to_datetime(originData['end_treatment_date'])
originData['CureDay'] = (originData['end_treatment_date'] - originData['beginning_of_treatment_date']).dt.days

# 가설: other_cancer 유무와 사망률 간 관계 분석
# 귀무가설: other_cancer와 생존 여부는 관련이 없다
# 대립가설: other_cancer와 생존 여부는 관련 있다

# 교차표
crossTable = pd.crosstab(originData['other_cancer'], originData['survived'])
print("교차표:\n", crossTable)

# 카이제곱 검정
chi2, p, dof, expected = chi2_contingency(crossTable)
print(f"\n카이제곱 통계량 = {chi2:.3f}, p-value = {p:.5f}")

alpha = 0.05
print('p값 : ', p)
if p < alpha:
    print("=> 귀무가설 기각: 다른 암 보유 여부는 생존과 관련 있음.")
else:
    print("=> 귀무가설 채택: 다른 암 보유 여부는 생존과 관련 없음.")
#결과 : 귀무가설 기각: 다른 암 보유 여부는 생존과 관련 있음,  p값 : 0.026173252356251984


# 시각화
sns.countplot(x='other_cancer', hue='survived', data=originData)
plt.title("다른 암 보유 여부에 따른 생존 분포")
plt.xlabel("다른 암 보유 여부 (0: 없음, 1: 있음)")
plt.ylabel("사람 수")
plt.legend(title='생존여부', labels=['사망(0)', '생존(1)'])
plt.tight_layout()
plt.savefig("다른 암 보유 여부에 따른 생존 분포.png")
plt.show()

# 머신러닝 모델링
# 결측치 제거
originData.dropna(inplace=True)

# 모델 특성 설정
data = originData[['other_cancer', 'survived']].copy()
feature = data[['other_cancer']]
label = data['survived']

# 훈련/테스트 분리
xTrain, xTest, yTrain, yTest = train_test_split(feature, label, test_size=0.2, random_state=1)

# 랜덤 포레스트 모델 학습
rfModel = RandomForestClassifier(n_estimators=100, random_state=42)
rfModel.fit(xTrain, yTrain)
pred = rfModel.predict(xTest)

# 모델 피클로 저장
with open('rf_model_other_cancer.pkl', 'wb') as file:
    pickle.dump(rfModel, file)

# 결과 출력
print("Confusion Matrix:\n", confusion_matrix(yTest, pred))
print("\nClassification Report:\n", classification_report(yTest, pred))

# 특성 중요도
importances = pd.Series(rfModel.feature_importances_, index=feature.columns).sort_values(ascending=False)
plt.figure(figsize=(6, 3))
sns.barplot(x=importances.values, y=importances.index)
plt.title("특성 중요도")
plt.tight_layout()
plt.savefig("특성중요도.png")
plt.show()

# 위험도 매트릭스 (다른 암 보유 여부별 사망률)
risk_matrix = originData.groupby('other_cancer')['survived'].apply(lambda x: (x == 0).mean())

plt.figure(figsize=(6, 2))
sns.heatmap(risk_matrix.to_frame().T, annot=True, cmap='Reds', cbar=True, fmt=".2f")
plt.title("다른 암 보유 여부별 사망률 위험도 매트릭스")
plt.xlabel("다른 암 보유 여부")
plt.yticks([])
plt.tight_layout()
plt.savefig("위험도_매트릭스.png")
plt.show()