#타이타닉 데이터 셋으로 로지스틱 리그래션, 디시전 트리 클래시파이어, 랜덤포레스트 클래시파이어 비교

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
print(df.head(2), df.shape)
print(df.describe())

# Null 처리 : 제거가 아닌 평균값 또는 'N'을 넣는 방법으로.
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)

print(df.isnull().sum())

# 
print(df.info())
print('Sex : ', df['Sex'].value_counts())
print('Cabin : ', df['Cabin'].value_counts())
print('Embarked : ', df['Embarked'].value_counts())
df['Cabin'] = df['Cabin'].str[:1]
print(df.head(3))

# 성별이 생존 확률에 어떤 영향을 미쳤는지 확인하기
print(df.groupby(['Sex', 'Survived'])['Survived'].count())
print('여성 생존률 : ', 233 / (81 + 233))   #0.7420382165605095
print('남성 생존률 : ', 109 / (468 + 109))  #0.18890814558058924
sns.barplot(x='Sex', y='Survived', data=df, errorbar=('ci', 95))
#plt.show()

# 성별 기준으로 Pclass별 생존 확률
sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = df)
#plt.show()
plt.close()

# 나이별 기준으로 생존 확률
def getAgeFunc(age) : 
    msg = ''
    if age <= -1: msg = 'unknown'
    elif age <= 5: msg = 'baby'
    elif age <= 18: msg = 'teenager'
    elif age <= 65: msg = 'adult'
    else: msg = 'elder'
    return msg

df['Age_category'] = df['Age'].apply(lambda a: getAgeFunc(a))
print(df.head(2))
sns.barplot(x='Age_category', y='Survived', hue='Sex', data=df, order=['unknown', 'baby', 'teenager', 'adult', 'elder'])
#plt.show()
plt.close()
del df['Age_category']

# 문자열 자료를 숫자화
from sklearn import preprocessing
def labelIncoder(datas) : 
    cols = ['Cabin', 'Sex', 'Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique())         #[7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique())           #[1 0]
print(df['Embarked'].unique())      #[3 0 2 1]


# feature / label
feature_df = df.drop(['Survived'], axis='columns')
label_df = df['Survived']
print(feature_df.head(2))
print(label_df.head(2))

x_train, x_test, y_trian, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=1)
print(x_train.shape, x_test.shape, y_trian.shape, y_test.shape) #(712, 8) (179, 8) (712,) (179,)
logiModel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_trian)
deceModel = DecisionTreeClassifier().fit(x_train, y_trian)
rfModel = RandomForestClassifier().fit(x_train, y_trian)

logPred = logiModel.predict(x_test)
print('logModel 정확도 : {0:.5f}'.format(accuracy_score(y_test, logPred)))  #logModel 정확도 : 0.79888
decePred = deceModel.predict(x_test)    
print('deceModel 정확도 : {0:.5f}'.format(accuracy_score(y_test, decePred)))#deceModel 정확도 : 0.73743
rfPred = rfModel.predict(x_test)
print('rfModel 정확도 : {0:.5f}'.format(accuracy_score(y_test, rfPred)))    #rfModel 정확도 : 0.75419