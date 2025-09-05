"""
KNN(최근접 이웃)

"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
xTrain, xTest, yTrain, yTest = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66) #stratify : 클래스 비율 평준화
train_acc = []
test_acc = []
neighbors_set = range(1, 11, 2) #1부터 11까지, 2씩 끊어서(과적합 방지를 위해)

for n_neighbor in neighbors_set:
    clf = KNeighborsClassifier(n_neighbors=n_neighbor, p=1, metric='minkowski')
    clf.fit(xTrain, yTrain)
    train_acc.append(clf.score(xTrain, yTrain))
    test_acc.append(clf.score(xTest, yTest))

import numpy as np
print('train 분류 정확도 평균 : ', np.mean(train_acc))
print('test 분류 정확도 평균 : ', np.mean(test_acc))

plt.plot(neighbors_set, train_acc, label = 'train_acc')
plt.plot(neighbors_set, test_acc, label= 'test_acc')
plt.ylabel('acc')
plt.xlabel('k')
plt.legend()
plt.show()