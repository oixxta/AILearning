"""
문제4) testdata/HR_comma_sep.csv 파일을 이용하여 salary를 예측하는 분류 모델을 작성한다.
* 변수 종류 *
satisfaction_level : 직무 만족도
last_evaluation : 마지막 평가점수
number_project : 진행 프로젝트 수
average_monthly_hours : 월평균 근무시간
time_spend_company : 근속년수
work_accident : 사건사고 여부(0: 없음, 1: 있음)
left : 이직 여부(0: 잔류, 1: 이직)
promotion_last_5years: 최근 5년간 승진여부(0: 승진 x, 1: 승진)
sales : 부서

salary : 임금 수준 (low, medium, high)

조건 : Randomforest 클래스로 중요 변수를 찾고, Keras 지원 딥러닝 모델을 사용하시오.
Randomforest 모델과 Keras 지원 모델을 작성한 후 분류 정확도를 비교하시오.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from tensorflow import keras
from keras import layers

#데이터 가져오기
url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/HR_comma_sep.csv'
data = pd.read_csv(url)
print(data.isnull().any())  #결측치 없음
print(data.columns) #['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']
print(data.shape)   #(14999, 10)

# salary 칼럼의 데이터 (low, medium, high)를 명목척도 범주형 데이터(0, 1, 2)로 바꿈.
encoder = LabelEncoder()
data.loc[:, 'salary'] = encoder.fit_transform(data['salary'])
print(data['salary'].value_counts())
"""
1    7316
2    6446
0    1237
"""

#피쳐와 레이블 나누기
xData = data.drop(columns=['salary'])
yData = data['salary'].values.ravel().astype(int)

xData = pd.get_dummies(xData, columns=['sales'], drop_first=False)

#학습데이터 나누기
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=0, stratify=yData)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(10499, 9) (4500, 9) (10499, 1) (4500, 1)

#모델 1번 : RandomForestClassifier
def modelRF():
    model = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=42, n_jobs=-1)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    print('예측값 : ', yPred[:10])               #예측값 :  [2 1 1 1 1 1 1 1 1 1]
    print('실제값 : ', np.array(yTest[:10]))     #실제값 :  [2 1 2 1 1 1 2 1 2 1]
    print('맞춘 갯수 : ', sum(yTest == yPred))   #맞춘 갯수 :  2764
    print('전체 대비 맞춘 비율 : ', sum(yTest == yPred) / len(yTest))   #0.6142222222222222
    print('분류 정확도 : ', accuracy_score(yTest, yPred))              #0.6142222222222222

    #중요변수 확인하기
    print('특성(변수) 중요도 : ', model.feature_importances_)

    #시각화로 중요변수 확인
    n_features = xData.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.xlabel('feature_importances score')
    plt.ylabel('features')
    plt.yticks(np.arange(n_features), xData.columns)
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()
    #불순도를 떨어트리는데 제일 큰 기여를 한 칼럼은 'average_monthly_hours'(월평균 근무시간)



num_cols = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'Work_accident',
    'left', 'promotion_last_5years'
]
cat_cols = [c for c in xData.columns if c.startswith('sales_')]  # get_dummies로 생긴 부서 원-핫

scaler = StandardScaler()
xTrain[num_cols] = scaler.fit_transform(xTrain[num_cols])
xTest[num_cols]  = scaler.transform(xTest[num_cols])

xTrain = xTrain.astype(np.float32).values
yTrain = yTrain.astype(np.int32)
xTest = xTest.astype(np.float32).values
yTest = yTest.astype(np.int32)


#모델 2번 : Keras : Sequential
def modelSQ():
    model = Sequential()
    model.add(Input(shape=(xTrain.shape[1],)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=3, activation='softmax'))
    print(model.summary())
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    earlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(xTrain, yTrain, epochs=1000, validation_split=0.2, callbacks=[earlyStop], verbose=1)

    loss, acc = model.evaluate(xTest, yTest, verbose=0)
    print(f'최종 평가 : Loss : {loss:.4f}, accuracy : {acc:.4f}')   #최종 평가 : Loss : 0.8809, accuracy : 0.5173

    #학습에 대한 곡선 시각화 해보기
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], '--', label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], '--', label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.close()

#모델 3번 : Keras : modelSubclassing
def modelSC():
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = layers.Dense(64, activation='relu')
            self.dropout1 = layers.Dropout(0.2)
            self.dense2 = layers.Dense(32, activation='relu')
            self.dropout2 = layers.Dropout(0.2)
            self.out = layers.Dense(3, activation='softmax')
        
        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            return self.out(x)
    
    model = MyModel()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    earlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(xTrain, yTrain, epochs=1000, validation_split=0.2, callbacks=[earlyStop], verbose=1)

    loss, acc = model.evaluate(xTest, yTest, verbose=0)
    print(f'최종 평가 : Loss : {loss:.4f}, accuracy : {acc:.4f}')   #최종 평가 : Loss : 0.8809, accuracy : 0.5173


#modelRF()
#modelSQ()
modelSC()