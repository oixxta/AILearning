# RandomForest 분류/예측 알고리즘
# 분류 알고리즘으로 타이타닉 데이터 세트 사용해 이진 분류
# Bagging 사용 : 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고, 결과를 집계(Aggregating)하는 방법.
# 참고 : 우수한 성능을 원한다면, Boosting, 오버피팅이 걱정된다면, Bagging 추천.

# titanic dataset : feature(pclass, age, sex), label(survived)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(5))
print(df.columns)   #['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
print(df.shape)     #(891, 12)
print(df.info())
print(df.isnull().any())    #결측치 확인
df = df.dropna(subset=['Pclass', 'Age', 'Sex']) #결측치 제거
print(df.shape)     #(714, 12)

# 독립변수 df 만들기 및 feature, label로 분리
df_x = df[['Pclass', 'Age', 'Sex']].copy()
print(df_x.head(2))

# sex 칼럼의 데이터 (male과 female)를 명목척도 범주형 데이터(0, 1)로 바꿈.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:, 'Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head(2))

# 종속변수
df_y = df['Survived']
print(df_y.head(2))

# 학습데이터 분리
trainX, testX, trainY, testY = train_test_split(df_x, df_y, test_size=0.3, random_state=12)

# 모델 생성
model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(trainX, trainY)
pred = model.predict(testX)
print('예측값 : ', pred[:10])               #예측값 :  [1 0 0 0 0 0 1 1 0 1]
print('실제값 : ', np.array(testY[:10]))    #실제값 :  [1 0 0 0 1 0 1 1 0 1]
print('맞춘 갯수 : ', sum(testY == pred))    #맞춘 갯수 :  180
print('전체 대비 맞춘 비율 : ', sum(testY == pred) / len(testY))    #전체 대비 맞춘 비율 :  0.8372093023255814
print('분류 정확도 : ', accuracy_score(testY, pred))               #분류 정확도 :  0.8372093023255814

# K-Fold
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))     #교차 검증 평균 정확도 :  0.81514

# 중요변수 확인
print('특성(변수) 중요도 : ', model.feature_importances_)      #특성(변수) 중요도 :  [0.15718966 0.55741765 0.28539269]

# 시각화로 중요변수의 기여도 확인
import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('feature_importances score')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()         #불순도(엔트로피)를 떨어트리는데 제일 큰 기여를 한 칼럼은 'Age'