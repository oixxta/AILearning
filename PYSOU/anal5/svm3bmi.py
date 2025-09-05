"""
SVM으로 분류모델 만들기

BMI 식을 이용해 데이터 만들기
비만도 계산은 몸무게를 키의 제곱으로 나눈 것
예) 키:175, 몸무게 68 ==> 68 / ((170 / 100) * (170 / 100))
"""
print(68 / ((170 / 100) * (170 / 100))) #23.529411764705884


import random
random.seed(12)

#BMI 계산기
def calc_bmi(height, weight):
    bmi = weight / (height / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 25.5: return 'normal'
    return 'faty'
print(calc_bmi(170, 68))    #23.529411764705884, normal

"""
#데이터 무작위 생성 후 저장
fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n')   #제목
count = {'thin':0, 'normal':0, 'faty':0}    
for i in range(50000):
    height = random.randint(150, 200)
    weight = random.randint(35, 100)
    label = calc_bmi(height, weight)
    count[label] += 1
    fp.write('{0},{1},{2}\n'.format(height, weight, label))

fp.close()
"""

#만든 데이터 읽어오기
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

table = pd.read_csv("bmi.csv")
print(table.head(3), table.shape)
"""
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   height  50000 non-null  int64
 1   weight  50000 non-null  int64
 2   label   50000 non-null  object

(50000, 3) 
"""
print(table.info())
"""
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   height  50000 non-null  int64
 1   weight  50000 non-null  int64
 2   label   50000 non-null  object
"""

label = table['label']
print(label[:3])

w = table['weight'] / 100       #정규화
print(w[:3].values)             #[0.69 0.79 0.83]
h = table['height'] / 200       #정규화
print(w[:3].values)             #[0.69 0.79 0.83]
wh = pd.concat([w, h], axis=1)
print(wh.head(3), wh.shape)
"""
   weight  height
0    0.69   0.900
1    0.79   0.960
2    0.83   0.795 (50000, 2)
"""
label = label.map({'thin' : 0, 'normal' : 1, 'faty' : 2})  #범주형 변수 label를 정수화
print(label[:3])
"""
0    1
1    1
2    2
"""

#학습/테스트 데이터 나누기
xTrain, xTest, yTrain, yTest = train_test_split(wh, label, test_size=0.3, random_state=1)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(35000, 2) (15000, 2) (35000,) (15000,)

#모델 만들기
model = svm.SVC(C=0.01, kernel='rbf', random_state=1).fit(xTrain, yTrain)  
#C= : 오류허용정도, 규제 강도. 숫자값이 커질수록 과적합이 될 우려가 있음.
#kernel : 차원을 고차원으로 높힘 (기본값 : rbf)

#모델 테스트
pred = model.predict(xTest)
print("예측값 : ", pred[:10])               #예측값 :  [2 0 1 1 0 0 2 1 0 0]
print("실제값 : ", yTest[:10].values)       #실제값 :  [2 0 1 1 0 0 2 1 0 0]
acScore = metrics.accuracy_score(yTest, pred)   
print('정확도 : ', acScore)                 #정확도 :  0.9736666666666667

#시각화 해보기
table2 = pd.read_csv("bmi.csv", index_col=2)
def scatterFunc(lbl, color):
    b = table2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)

scatterFunc('faty', 'red')
scatterFunc('normal', 'yellow')
scatterFunc('thin', 'blue')
plt.legend()
plt.show()
plt.close()

#새로운 값 예측하기
newData = pd.DataFrame({'weight' : [69, 89], 'height' : [170, 170]})
newData['weight'] = newData['weight'] / 100 #학습 모델이 정규화를 시켰기에, 이 것도 정규화 해야 함.
newData['height'] = newData['height'] / 200
newPred = model.predict(newData)
print('새로운 데이터에 대한 bmi : ', newPred)   # [1 1]