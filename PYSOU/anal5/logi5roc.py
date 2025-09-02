"""
분류모델 성능평가 관련


"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
print(x[:3])    #[[-0.01032243 -0.80566819] [-1.10293659  2.21661117] [-1.90795358 -0.20839902]]
print(y[:3])    #[1 0 0]

#import matplotlib.pyplot as plt
#plt.scatter(x[:, 0], x[:, 1])
#plt.show()

model = LogisticRegression().fit(x, y)
yHat = model.predict(x)            #               [1 0 0]
print('예측값(yHat) : ', yHat[:3])  #예측값(yHat) :  [0 0 0], 2개 맞추고 한개 틀림.


f_value = model.decision_function(x)    #결정함수(판별함수, 불확실성 추정 함수), 판별경계선 설정을 위한 샘플 자료 얻기
print('f_value : ', f_value[:10])   #[-0.28579821 -0.94089633 -4.23246754  2.80425793  0.06137063  2.88531461 -2.1095613  -1.76266592 -0.93092321  2.98569333]
df = pd.DataFrame(np.vstack([f_value, yHat, y]).T, columns=['f', 'yHat', 'y'])
print(df.head(3))
"""
           f  yHat    y
0  -0.285798   0.0  1.0
1  -0.940896   0.0  0.0
2  -4.232468   0.0  0.0
3   2.804258   1.0  1.0
4   0.061371   1.0  0.0
..       ...   ...  ...
95  3.089143   1.0  1.0
96 -4.588386   0.0  0.0
97  1.542359   1.0  0.0
98  0.604240   1.0  1.0
99  5.459643   1.0  1.0
"""

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, yHat))
"""
[[44  4]
 [ 8 44]]   맞춘 갯수 : 88개, 틀린 갯수 12개. 정확도 88%
"""
acc = (44 + 44) / 100           #정확도
recall = 44 / (44 + 4)          #재현률
precision = 44 / (44 + 8)       #정밀도 : 양성 가운데 맞춘 양성의 수
specificity = 44 / (8 + 44)     #특이도 : 음성 가운데 맞춘 음성의 수   TN / FP + TN
fallout = 8 / (8 + 44)          #위양성률 FP / (FP + TN)

print('acc(정확도) : ', acc)                    #0.88
print('recall(재현률) : ', recall)              #0.9166666666666666
print('precision(정밀도) : ', precision)        #0.8461538461538461
print('specificity(특이도) : ', specificity)    #0.8461538461538461
print('fallout(위양성률) : ', fallout)          #0.15384615384615385
print('fallout(위양성률) : ', 1 - specificity)  #0.15384615384615385
# 정리하면 TPR은 1에 근사하면 좋고, FPR은 0에 근사하면 좋다.

from sklearn import metrics
ac_score = metrics.accuracy_score(y, yHat)
print('ac_score : ', ac_score)                  #0.88
cl_rep = metrics.classification_report(y, yHat)
print('cl_rep : ', cl_rep)
"""
                precision    recall  f1-score   support

           0       0.85      0.92      0.88        48           클래스 0
           1       0.92      0.85      0.88        52           클래스 1

    accuracy                           0.88       100
   macro avg       0.88      0.88      0.88       100           #macro avg : 가중치 평균값
weighted avg       0.88      0.88      0.88       100           #weighted avg : 샘플이 많은 클래스의 성능이 더 많이 반영된 가중평균값
"""
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류임계결정값 : ', thresholds)
"""
fpr :  [0.         0.         0.         0.02083333 0.02083333 0.04166667
 0.04166667 0.10416667 0.10416667 0.14583333 0.14583333 0.27083333
 0.27083333 0.29166667 0.29166667 0.41666667 0.41666667 0.45833333
 0.45833333 0.47916667 0.47916667 1.        ]
tpr :  [0.         0.01923077 0.78846154 0.78846154 0.82692308 0.82692308
 0.84615385 0.84615385 0.88461538 0.88461538 0.90384615 0.90384615
 0.92307692 0.92307692 0.94230769 0.94230769 0.96153846 0.96153846
 0.98076923 0.98076923 1.         1.        ]
분류임계결정값 :  [        inf  6.0464992   1.67684354  1.54235863  0.88283652  0.61223489
  0.60424048 -0.01703242 -0.31365278 -0.72227484 -0.73423846 -1.03471153
 -1.06352271 -1.12845873 -1.26884667 -1.50495489 -1.58822082 -1.70988114
 -1.76266592 -1.77066806 -1.81724936 -5.91183146]
"""

# ROC 커브 시각화 하기
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'o-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier line')
plt.plot([fallout], [recall], 'ro', ms=10)      #위양성률과 재현율 값 출력하기
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()
plt.close()

# AUC(Area Under the Curve) - ROC 커브의 면적
# 1에 가까울 수록 좋은 분류모델로 평가됨.
print('AUC : ', metrics.auc(fpr, tpr))  #AUC :  0.9547275641025641
