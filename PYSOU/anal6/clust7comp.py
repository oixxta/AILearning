"""
iris 데이터 세트로 지도학습과 비지도학습 두 가지 비교해보기.

지도학습 : KNN, 비지도학습 : K-means 사용.

결정적인 차이점 : label 필요 유무
"""

#데이터 가져오기 및 타입 확인
from sklearn.datasets import load_iris

iris_Dataset = load_iris()
print(iris_Dataset.keys())      
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
print(iris_Dataset['data'][:3])
print(iris_Dataset['feature_names'])    #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris_Dataset['target'][:3])
print(iris_Dataset['target_names'][:3]) #['setosa' 'versicolor' 'virginica']


#학습 데이터 나누기
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(iris_Dataset['data'], iris_Dataset['target'], test_size=0.25, random_state=42)
print(trainX.shape, testX.shape, trainY.shape, testY.shape) #(112, 4) (38, 4) (112,) (38,)


#지도학습 : KNN
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
knnModel = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
#knnModel = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='minkowski')
knnModel.fit(trainX, trainY)    #feature, label(tag, target, class) 순으로.
predict_label = knnModel.predict(testX)
print('예측값 : ', predict_label)
print('정확도 : {:3f}'.format(np.mean(predict_label == testY)) )    #정확도 : 1.000000
print('acc : ', metrics.accuracy_score(testY, predict_label))      #acc :  1.0
#새로운 데이터 분류
newInput = np.array([[6.1, 2.8, 4.7, 1.2]])
print(knnModel.predict(newInput))
print(knnModel.predict_proba(newInput))
dist, index = knnModel.kneighbors(newInput)
print(dist, index)


#비지도학습 : K-means (데이터에 정답(label, 종속변수)이 없는경우)
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeansModel.fit(trainX)     #feature만 필요. 에초에 feature밖에 없을 때 사용.
#print(kmeansModel.labels_)
#[1 1 2 2 2 1 1 2 2 0 2 0 2 0 2 1 0 2 1 1 1 2 2 1 1 1 2 1 2 0 1 2 2 1 2 2 2
# 2 0 2 1 2 0 1 1 2 0 1 2 1 1 2 2 0 2 0 0 2 1 1 2 0 1 1 1 2 0 1 0 0 1 2 2 2
# 0 0 1 0 2 0 2 2 2 1 2 2 1 2 0 0 1 2 0 0 1 0 1 0 0 0 2 0 2 2 2 2 1 2 2 1 2
# 0]
print('0번째 클러스터 : ', trainY[kmeansModel.labels_ == 0])
print('1번째 클러스터 : ', trainY[kmeansModel.labels_ == 1])
print('2번째 클러스터 : ', trainY[kmeansModel.labels_ == 2])
#0번째 클러스터 :  [2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2]
#1번째 클러스터 :  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
#2번째 클러스터 :  [2 1 1 1 2 1 1 1 1 1 2 1 1 1 2 2 2 1 1 1 1 1 2 1 1 1 1 2 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 2 2 1 2 1]
#이번엔 클러스터링에서 새로운 데이터 분류
newInput = np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(newInput)
print(clu_pred)         # [2]


