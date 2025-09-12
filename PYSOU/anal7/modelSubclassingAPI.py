import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#공부시간에 따른 성적 데이터 사용
xData = np.array([1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)
#xData = [[1], [2], [3], [4] ,[5]]와 같음.
yData = np.array([11, 32, 53, 64, 70], dtype=np.float32).reshape(-1, 1)

class MyModel(Model):
    def __init__(self):                             # 케라스 Model을 상속받아 사용자 정의 모델 클래스 선언
        super(MyModel, self).__init__()             # 생성자: 레이어 정의 및 하이퍼파라미터 초기화 위치
        self.d1 = Dense(16, activation='relu')      #히든레이어 : 노드 16개, ReLU 활성화 함수 사용
        self.d2 = Dense(1, activation='linear')     #출력층 : 노드 1개, 회귀용 선형 출력(예: y의 연속값 예측)

    # x는 input 매개변수 : 함수형 API와 유사하나, Input 객체를 생성하진 않음.
    # 계산 작업 등을 할 수 있다.
    # model.fit(), evaluate(), predict() 등을 하면 자동으로 호출됨.
    def call(self, x):  
        x = self.d1(x)      #input 객체를 만들어서 히든레이어에 넣음.
        return self.d2(x)   #출력층의 결과를 반환함.

model3 = MyModel()          # 위에서 정의한 사용자 모델 인스턴스화

opti3 = optimizers.SGD(learning_rate=0.001) # 확률적 경사하강법(SGD) 옵티마이저 생성(학습률 0.001)
model3.compile(optimizer=opti3, loss='mse', metrics=['mse'])    # 모델 학습 준비(손실함수/옵티마이저/평가지표 설정)

history2 = model3.fit(x=xData, y=yData, batch_size=1, epochs=100, verbose=2)    # 모델 학습 시작
loss_metrics = model3.evaluate(x=xData, y=yData, verbose=0)     # 학습된 모델 성능 평가(손실/지표 반환)
print('loss_metrics : ', loss_metrics)          # 평가 결과 출력: [loss, mse] 순서의 리스트

#성능확인
yPred = model3.predict(xData, verbose=0)    # 학습된 모델로 예측값 산출
print('실제값 : ', yData.ravel())           
print('예측값 : ', yPred.ravel())           
print('설명력 : ', r2_score(yData, yPred))

"""
서브클래싱 언제 쓰나?
입력/출력 구조가 유동적이거나, call() 내부에서 조건문/반복문/맞춤 계산을 자유롭게 쓰고 싶을 때 탁월. 
(예: 여러 입력을 상황에 따라 다르게 합치기, 시뮬레이션식 계산 등)

배치 사이즈
batch_size는 실제 학습에선 16/32 등 조금 더 큰 값을 씀.(속도/안정성 향상).

출력층 활성화
회귀에서는 보통 activation='linear'를 씀. 분류라면 마지막 층과 손실함수를 목적에 맞게 바꿔야 함.
(예: 이진분류→ Dense(1, 'sigmoid') + binary_crossentropy, 
다중분류→ Dense(C, 'softmax') + categorical_crossentropy/sparse_categorical_crossentropy).

데이터 전처리
입력은 보통 스케일 정규화(표준화/정규화)를 하면 학습이 훨씬 안정적. 타깃도 범위가 크면 스케일링을 고려.

평가지표
MSE 외에 mae(평균절대오차)도 직관적입니다. compile(..., metrics=['mse','mae'])처럼 여러 개 넣어도 됨..
"""