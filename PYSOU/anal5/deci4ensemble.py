"""
앙상블(Ensemble)
하나의 샘플 데이터를 여러 개의 분류기를 통해 다수의 학습모델을 만들어 학습시키고,
학습 결과를 결합함으로써 과적합을 방지하고, 정확도를 높이는 학습 기법이다.

엉성불의 보팅 방법으로는 하드보팅과 소프트보팅 두 가지 방법으로 나뉨.
하드보팅은 예측한 결괏값들중 다수의 분류기가 결정함.
소프트보팅은 확률의 평균을 내서 최종 결정을 함. 보통은 소프트보팅의 성능이 더 좋아서 소프트보팅을 씀.

위스콘신 대학교의 유방암 데이터
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import numpy as np

cancerData = load_breast_cancer()
x, y = cancerData.data, cancerData.target
print(x[:2])
print(y[:2])
print(np.unique(y)) # 0:악성, 암 / 1:양성, 암이 아님.

#0과 1의 비율 확인
counter = Counter(y)
total = sum(counter.values())
for cls, cnt in counter.items():
    print(f"class {cls} : {cnt}개 ({cnt / total})") #class 0 : 212개 (0.37258347978910367), class 1 : 357개 (0.6274165202108963)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=12, stratify=y)
# stratify : 레이블 분포가 train / test 고르게 유지하도록 층화 샘플링
# 불균형 데이터에서 모델 평가가 왜곡되지 않도록 함.
from collections import Counter
y_li = y.tolist()
ytr_li = yTrain.tolist()
yte_li = yTest.tolist()

print('전체 분포 : ', Counter(y_li))               #Counter({np.int64(1): 357, np.int64(0): 212})
print('train 분포 : ', Counter(ytr_li))        #Counter({np.int64(1): 285, np.int64(0): 170})
print('test 분포 : ', Counter(yte_li))          #Counter({np.int64(1): 72, np.int64(0): 42})

# 개별 모델 생성 (스케일링-표준화)
# make_pipeline을 이용해 전처리와 모델을 일체형으로 관리.
logi = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000, random_state=12))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
tree = DecisionTreeClassifier(max_depth=5, random_state=12)
voting = VotingClassifier(estimators=[('LR', logi), ('KNN', knn), ('DT', tree)], voting='soft')
#서로다른 3개의 분류기를 voting에서 하나로 합쳤음.

#개별 모델 성능들을 확인.
for clf in [logi, knn, tree]:
    clf.fit(xTrain, yTrain)
    pred = clf.predict(xTest)
    print(f'{clf.__class__.__name__} 정확도 : {accuracy_score(yTest, pred):.4f}')
"""
전체 분포 :  Counter({np.int64(1): 357, np.int64(0): 212})
train 분포 :  Counter({np.int64(1): 285, np.int64(0): 170})
test 분포 :  Counter({np.int64(1): 72, np.int64(0): 42})
Pipeline 정확도 : 0.9912
Pipeline 정확도 : 0.9737
DecisionTreeClassifier 정확도 : 0.8772
"""
#name_models = [('LR', logi), ('KNN', knn), ('DT', tree)]
#for name, clf in name_models:
#    clf.fit(xTrain, xTest)
#    pred = clf.predict(xTest)
#    print(f'{name} 정확도 : {accuracy_score(yTest, pred):.4f}')



voting.fit(xTrain, yTrain)
vpred = voting.predict(xTest)
print(f'voting 분류기 정확도 : {accuracy_score(yTest, vpred):.4f}') #voting 분류기 정확도 : 0.9649

# 옵션 : 교차검증으로 안정성 확인 해보기.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
cv_score = cross_val_score(voting, x, y, cv=cv, scoring='accuracy')
print(f'voting 5겹 cv 평균 : {cv_score.mean():.4f} (+-) {cv_score.std():.4f}')  #voting 5겹 cv 평균 : 0.9701 (+-) 0.0181

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(yTest, vpred, digits=4))
"""
              precision    recall  f1-score   support

           0     0.9524    0.9524    0.9524        42
           1     0.9722    0.9722    0.9722        72

    accuracy                         0.9649       114
   macro avg     0.9623    0.9623    0.9623       114
weighted avg     0.9649    0.9649    0.9649       114
"""
print(confusion_matrix(yTest, vpred))
#[[40  2]
# [ 2 70]]
print(roc_auc_score(yTest, voting.predict_proba(xTest)[:, 1]))      #0.994047619047619


# GridSearchCV로 하이퍼 파리미터(최적의 파라미터 찾기)
from sklearn.model_selection import GridSearchCV
param_grid = {
    'LR__logisticregression__C' : [0.1, 1.0, 10.0],
    'KNN__kneighborsclassifier__n_neighbors' : [3, 5, 7],
    'DT__max_depth' : [3, 5, 7]
}

gs = GridSearchCV(voting, param_grid, cv = cv, scoring='accuracy')
gs.fit(xTrain, yTrain)
print('best params : ', gs.best_params_)    #best params :  {'DT__max_depth': 5, 'KNN__kneighborsclassifier__n_neighbors': 5, 'LR__logisticregression__C': 1.0}
print('best cv accuracy : ', gs.best_score_)
best_voting = gs.best_estimator_       #best cv accuracy :  0.9714285714285715
print('test accuracy(best) : ', accuracy_score(yTest, best_voting.predict(xTest)))  #test accuracy(best) : 0.9649122807017544
print('test ROC-AUC(best) : ', roc_auc_score(yTest, best_voting.predict_proba(xTest)[:, 1]))    #test ROC-AUC(best) : 0.994047619047619