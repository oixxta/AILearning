"""
단층 신경망(뉴런, 노드) - 퍼셉트론(Perceptron) : 
input의 가중치 합에 대해 임계값을 기준으로 두 가지 output 중 하나를 출력하는 간단한 구조다.(이항분류 가능)

단층 신경망으로 논리회로 분류

y = w1 * x1 + w2 * x2 + .... + b
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
print(feature)                  #[[0 0] [0 1] [1 0] [1 1]]
label = np.array([0, 0, 0, 1])  # and연산
#and연산 학습 1회
ml = Perceptron(max_iter=1, eta0=0.1).fit(feature, label)  # max_iter : ephoc 횟수 지정, eta0 : learning rate. 학습률(학습량)
print(ml)
pred = ml.predict(feature)
print('예측결과 : ', pred)                      #예측결과 :  [0 0 0 0]
print('실제값 : ', label)                       #실제값 :  [0 0 0 1]
print('정확도 : ', accuracy_score(label, pred)) #정확도 :  0.75
#and연산 학습 100회
ml2 = Perceptron(max_iter=100, eta0=0.1).fit(feature, label)
print(ml2)
pred = ml2.predict(feature)
print('예측결과 : ', pred)                      #예측결과 :   [0 0 0 1]
print('실제값 : ', label)                       #실제값 :  [0 0 0 1]
print('정확도 : ', accuracy_score(label, pred)) #정확도 :  1.0
# and 연산 결과는 퍼셉트론이 충분이 예측할 수 있음!


#or연산 학습 100회
label = np.array([0, 1, 1, 1])  # or연산
ml3 = Perceptron(max_iter=100, eta0=0.1).fit(feature, label)
print(ml3)
pred = ml3.predict(feature)
print('예측결과 : ', pred)                      #예측결과 :   [0 1 1 1]
print('실제값 : ', label)                       #실제값 :  [0 1 1 1]
print('정확도 : ', accuracy_score(label, pred)) #정확도 :  1.0
# or 연산 결과는 퍼셉트론이 충분이 예측할 수 있음!


#xor 연산 학습 1000회
label = np.array([0, 1, 1, 0])  # xor연산
ml4 = Perceptron(max_iter=100, eta0=0.1).fit(feature, label)
print(ml4)
pred = ml4.predict(feature)
print('예측결과 : ', pred)                      #예측결과 :   [0 0 0 0]
print('실제값 : ', label)                       #실제값 :  [0 1 1 0]
print('정확도 : ', accuracy_score(label, pred)) #정확도 :  0.5
# 단층신경망(단층 퍼셉트론)은 그래프상에서 직선으로 xor 결과를 나눌 수 없기에, xor 연산을 할 수 없음!


