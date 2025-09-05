"""
다층 신경망(뉴런, 노드) - 퍼셉트론(Multi Layer Perceptron) : 

다층 신경망으로 논리회로 분류

y = w1 * x1 + w2 * x2 + .... + b
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

feature = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
print(feature)                  #[[0 0] [0 1] [1 0] [1 1]]
label = np.array([0, 1, 1, 0])  # xor연산
#xor연산 학습 1회
#ml = MLPClassifier(hidden_layer_sizes=30, solver='adam', learning_rate_init=0.01).fit(feature, label)
ml = MLPClassifier(hidden_layer_sizes=(10, 10, 10), solver='adam', learning_rate_init=0.01).fit(feature, label)
print(ml)
pred = ml.predict(feature)
print('예측결과 : ', pred)                      #예측결과 :  [0 1 1 0]
print('실제값 : ', label)                       #실제값 :  [0 1 1 0]
print('정확도 : ', accuracy_score(label, pred)) #정확도 :  1.0