"""
SVM : 비확률적 이진 선형분류 모델 작성 가능.(3차원 이상의 고차원 지원.)
직선적인(선형) 분류 뿐만 아니라, 커널 트릭을 이용해 비선형 분류도 가능함.
커널(Kernels) : 선형 분류가 어려운 저차원 자료를 고차원 공간으로 매핑해서 분류.

LogisticRegression과 SVM으로 XOR 연산 처리 결과 분류 가능 비교해 보기.

"""
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
import pandas as pd
import numpy as np

xData = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]
xDf = pd.DataFrame(xData)
feature = np.array(xDf.iloc[:, 0:2])
label = np.array(xDf.iloc[:, 2])
print(feature)  #[[0 0] [0 1] [1 0] [1 1]], feature은 2차원
print(label)    #[0 1 1 0], label은 1차원

#LogisticRegression으로는 이항분류 불가능!
model = LogisticRegression()
model.fit(feature, label)
pred = model.predict(feature)
print('예측값 : ', pred)    #[0 0 0 0]
print('실제값 : ', label)   #[0 1 1 0]
print('분류정확도 : ', metrics.accuracy_score(label, pred)) #분류정확도 :  0.5, 50%

#SVM으로는 다차원이 가능하기에 같은 선형임에도 불구하고, 이항 분류가 가능!
model = svm.SVC()
model.fit(feature, label)
pred = model.predict(feature)
print('예측값 : ', pred)    #[0 1 1 0]
print('실제값 : ', label)   #[0 1 1 0]
print('분류정확도 : ', metrics.accuracy_score(label, pred)) #분류정확도 :  1.0, 100%