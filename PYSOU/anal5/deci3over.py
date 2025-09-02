"""
전통적 모델(비 딥러닝) 과적합 방지 처리 방법 : train/test split, KFold, GridSearchCV ...

"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
import numpy as np

iris = load_iris()
print(iris.keys())  #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
trian_data = iris.data
train_label = iris.target
print(trian_data[:3])
print(train_label[:3])


# 일반적인 분류 모델
dt_clf = DecisionTreeClassifier()
print(dt_clf)
dt_clf.fit(trian_data, train_label)
pred = dt_clf.predict(trian_data)
print('예측값 : ', pred)                #0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
print('실제값 : ', train_label)         #0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
print('분류 정확도 : ', accuracy_score(train_label, pred))      # 1.0, 100%, 전형적인 오버피팅임.
#과적합이 발생.

# 과적합 방지 방법 1 : train/test로 데이터를 나누기
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, shuffle=True, random_state=121)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
dt_clf.fit(x_train, y_train)   # train으로 학습
pred2 = dt_clf.predict(x_test) # test로 예측.
print('예측값 : ', pred2[:10])      #[1 2 1 0 0 1 1 1 1 2]
print('실제값 : ', y_test[:10])     #[1 2 1 0 0 1 1 1 1 2]
print('정확도 : ', accuracy_score(y_test, pred2))   #0.9555555555555556, 높은 정확도이면서도 오버피팅은 잘 방지됨.
#과적합이 해소 - 일반화된 모델, 포용성이 있는 모델이 생성됨.

# 과적합 방지 방법 2 : 교차검증(cross validation)
# K-Fold 교차 검증이 가장 일반적임.
# train dataset에 대해 k개의 data fold set을 만들어 K번 만큼 학습 도중에 검증 평가를 수행.
features = iris.data
labels = iris.target
dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=123)
kFold = KFold(n_splits=5)
cv_acc = []
print('iris shape : ', features.shape)
n_iter = 0
for train_index, test_index in kFold.split(features):
    #print('n_iter : ', n_iter)
    #print('train_index : ', len(train_index))
    #print('test_index : ', len(test_index))
    #n_iter += 1
    # kFold.split으로 변환된 인덱스를 이용해 학습용, 검증용 데이터 추출
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = labels[train_index], labels[test_index]
    #학습 및 예측
    dt_clf.fit(x_train, y_train)   #학습
    pred = dt_clf.predict(xtest)   #테스트
    n_iter += 1
    #반복 할때마다 정확도 확인하기
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수 : {0}, 교차검증 정확도 : {1}, 학습데이터 수 : {2}, 검증데이터 수 : {3}'.format(n_iter, acc, train_size, test_size))
    print('반복수 : {}, 검증인덱스 : {}'.format(n_iter, test_index))
    cv_acc.append(acc) #평균을 구하기 위해
print('평균 검증 정확도 : ', np.mean(cv_acc))
"""
반복수 : 1, 교차검증 정확도 : 1.0, 학습데이터 수 : 120, 검증데이터 수 : 30
반복수 : 1, 검증인덱스 : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29]
반복수 : 2, 교차검증 정확도 : 1.0, 학습데이터 수 : 120, 검증데이터 수 : 30
반복수 : 2, 검증인덱스 : [30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
 54 55 56 57 58 59]
반복수 : 3, 교차검증 정확도 : 0.967, 학습데이터 수 : 120, 검증데이터 수 : 30
반복수 : 3, 검증인덱스 : [60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
 84 85 86 87 88 89]
반복수 : 4, 교차검증 정확도 : 0.967, 학습데이터 수 : 120, 검증데이터 수 : 30
반복수 : 4, 검증인덱스 : [ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119]
반복수 : 5, 교차검증 정확도 : 1.0, 학습데이터 수 : 120, 검증데이터 수 : 30
반복수 : 5, 검증인덱스 : [120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
 138 139 140 141 142 143 144 145 146 147 148 149]
평균 검증 정확도 :  0.9868
"""
# StratifiedKFold : 왜곡된 데이터, 불균형한 분포를 가진 데이터 집합을 위한 K-fold 방식.
# ex)대출사기, 스팸메일, 강우량, 코로나 백신 검사...
from sklearn.model_selection import StratifiedKFold

skFold = StratifiedKFold(n_splits=5)
cv_acc = []
print('iris shape : ', features.shape)
n_iter = 0
for train_index, test_index in skFold.split(features, labels):      #skFold.split은 파라미터가 두 개임.
    # kFold.split으로 변환된 인덱스를 이용해 학습용, 검증용 데이터 추출
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = labels[train_index], labels[test_index]
    #학습 및 예측
    dt_clf.fit(x_train, y_train)   #학습
    pred = dt_clf.predict(xtest)   #테스트
    n_iter += 1
    #반복 할때마다 정확도 확인하기
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('1111반복수 : {0}, 교차검증 정확도 : {1}, 학습데이터 수 : {2}, 검증데이터 수 : {3}'.format(n_iter, acc, train_size, test_size))
    print('1111반복수 : {}, 검증인덱스 : {}'.format(n_iter, test_index))
    cv_acc.append(acc) #평균을 구하기 위해
print('평균 검증 정확도 : ', np.mean(cv_acc))


# 과적합 방지 방법 2 : 교차검증(cross validation)
# cross_val_score 함수 사용.
print('교차 검증 함수로 처리 ---')
data = iris.data
label = iris.target
score = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=5)
print('교차 검증별 분류 정확도 : ', np.round(score, 2))     #교차 검증별 분류 정확도 :  [0.97 0.97 0.9  0.93 1.  ]
print('평균 검증 정확도 : ', np.round(np.mean(score), 2))  #평균 검증 정확도 :  0.95


# 과적합 방지 방법 3 : GridSearchCV - 최적의 파라미터(hyper parameter)를 제공.
parameters = {'max_depth' : [1, 2, 3], 'min_samples_split' : [2, 3]}    #dict type
gird_dtree = GridSearchCV(dt_clf, param_grid=parameters, cv=3, refit=True)  #refit을 주면 재학습을 실시함.
gird_dtree.fit(x_train, y_train)    #자동으로 복수의 내부 모형을 생성 실행해 가며 최적의 파라미터 찾기를 함.

import pandas as pd
scoreDf = pd.DataFrame(gird_dtree.cv_results_)
#pd.set_option('display.max_columns', None)
print(scoreDf)
"""
   mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
0       0.000579      0.000054         0.000341        0.000099
1       0.000587      0.000110         0.000295        0.000048
2       0.000687      0.000105         0.000341        0.000064
3       0.000481      0.000009         0.000266        0.000008
4       0.000477      0.000003         0.000256        0.000002
5       0.000474      0.000004         0.000258        0.000006

   param_max_depth  param_min_samples_split  \
0                1                        2
1                1                        3
2                2                        2
3                2                        3
4                3                        2
5                3                        3

                                     params  split0_test_score  \
0  {'max_depth': 1, 'min_samples_split': 2}           0.657143
1  {'max_depth': 1, 'min_samples_split': 3}           0.657143
2  {'max_depth': 2, 'min_samples_split': 2}           0.942857
3  {'max_depth': 2, 'min_samples_split': 3}           0.942857
4  {'max_depth': 3, 'min_samples_split': 2}           0.971429
5  {'max_depth': 3, 'min_samples_split': 3}           0.971429

   split1_test_score  split2_test_score  mean_test_score  std_test_score  \
0           0.657143           0.657143         0.657143        0.000000
1           0.657143           0.657143         0.657143        0.000000
2           0.914286           0.942857         0.933333        0.013469
3           0.914286           0.942857         0.933333        0.013469
4           0.914286           0.942857         0.942857        0.023328
5           0.914286           0.942857         0.942857        0.023328

   rank_test_score
0                5
1                5
2                3
3                3
4                1
5                1
"""
print('best parameter : ', gird_dtree.best_params_)     # {'max_depth': 3, 'min_samples_split': 2}
print('best accuracy : ', gird_dtree.best_score_)       # 0.9428571428571427
#최적의 파라미터를 탑재한 모델이 제공됨!
estimator = gird_dtree.best_estimator_      #최적의 추정계가 탑재된 모델
pred = estimator.predict(x_test)
print('예측값 : ', pred)                # [1 2 1 0 0 1 1 1 1 2 2 1 1 0 0 2 1 0 2 0 2 2 1 1 1 1 0 0 2 2 1 2 0 0 1 2 0 0 0 2 2 2 2 0 1]
print('테스트데이터 정확도 : ', accuracy_score(y_test, pred))   # 0.9555555555555556
