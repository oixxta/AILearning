"""
차원 축소(PCA - 주성분 분석) : 성격이 비슷한 두 개의 독립형 변수를 하나의 변수로 합쳐줌.

"""

import numpy as np
import pandas as pd

#독립변수(Feature)
x1 = [95, 91, 66, 94, 68]
x2 = [56, 27, 25, 1, 9]
x3 = [57, 34, 9, 79, 4]
x = np.stack((x1, x2, x3), axis=0)      #2차원 변수 3개
print(x)

x = pd.DataFrame(x.T, columns=['x1', 'x2', 'x3'])
print(x)
"""
   x1  x2  x3
0  95  56  57
1  91  27  34
2  66  25   9
3  94   1  79
4  68   9   4
"""

#표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
print(x_std)    #표준화한 값
"""
[[ 0.9396856   1.71373408  0.71720807]
 [ 0.63159196  0.17983629 -0.09140887]
 [-1.29399328  0.07405024 -0.97034034]
 [ 0.86266219 -1.19538242  1.49066776]
 [-1.13994646 -0.7722382  -1.14612663]]
"""
print(scaler.inverse_transform(x_std))  #원상복귀

#PCA 처리
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print('PCA 처리')
print(pca.fit_transform(x_std))     #주성분 처리 결과(변수가 3개에서 2개로 줄어듬, 차원축소)
"""
[[ 1.56726527  1.36284445]
 [ 0.42684029  0.17290638]
 [-1.53635755  0.42759045]
 [ 1.30017221 -1.6363821 ]
 [-1.75792022 -0.32695918]]
"""
print(pca.inverse_transform(pca.fit_transform(x_std)))  #주성분 처리 되돌리기(표준화한 값으로 돌아왔지만, 일부 왜곡이 발생함.)
"""
[[ 1.04190149  1.68625965  0.62018956]
 [ 0.29202373  0.27110824  0.23089328]
 [-1.09487099  0.02052851 -1.15933782]
 [ 0.97968624 -1.22683711  1.37959405]
 [-1.21874047 -0.75105929 -1.07133907]]
"""
print(scaler.inverse_transform(pca.inverse_transform(pca.fit_transform(x_std))))    #원상복귀(했지만, 일부 데이터는 일어버림.)
"""
[[96.32707571 55.48056608 54.24044153]
 [86.59136335 28.72559514 43.16744273]
 [68.58521773 23.98811367  3.62422572]
 [95.51933099  0.40531498 75.84066052]
 [66.97701221  9.40041012  6.1272295 ]]
"""

#와인 데이터로 분류하기 연습 : PCA 전과 후 비교
print('와인 데이터로 분류(RandomForest) 연습하기\n')
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd
datas = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv", header=None)
print(datas.head(3))
"""
    0     1     2    3      4     5     6       7     8     9    10  11  12
0  7.4  0.70  0.00  1.9  0.076  11.0  34.0  0.9978  3.51  0.56  9.4   5   1
1  7.8  0.88  0.00  2.6  0.098  25.0  67.0  0.9968  3.20  0.68  9.8   5   1
2  7.8  0.76  0.04  2.3  0.092  15.0  54.0  0.9970  3.26  0.65  9.8   5   1
"""
#['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
#12번째가 퀄리티, 나머지 요소가 12번의 값을 결정함.

x = np.array(datas.iloc[:, 0:12])
y = np.array(datas.iloc[:, 12])
print(x[:2])
"""
[[ 7.4     0.7     0.      1.9     0.076  11.     34.      0.9978  3.51 0.56    9.4     5.    ]
 [ 7.8     0.88    0.      2.6     0.098  25.     67.      0.9968  3.2 0.68    9.8     5.    ]]
"""
print(y[:2], set(y))    # 1 : RED, 0 : White
"""
[1 1]
"""
trainX, testX, trainY, testY = train_test_split(x, y, random_state=1, test_size=0.3)
print(trainX.shape, testX.shape, trainY.shape, testY.shape) #(4547, 12) (1950, 12) (4547,) (1950,)

model = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(trainX, trainY)
pred = model.predict(testX)
print('예측값 : ', pred[:5])
print('정확도 : ', sklearn.metrics.accuracy_score(testY, pred)) #0.9933333333333333, 정확도가 지나치게 높음, 오버피팅

# PCA화 해서 다시 학습해보기
pca = PCA(n_components=3)   #주성분분석 : 12개의 변수를 3개로 줄임
xPca = pca.fit_transform(x)
print(xPca[:3])
"""
[[-8.41107009e+01 -1.53071850e-01  3.36053930e-02]
 [-4.87789853e+01  5.83931247e+00 -8.54085420e-01]
 [-6.37341298e+01 -8.84221579e-01 -4.15690922e-01]]
"""
trainX, testX, trainY, testY = train_test_split(xPca, y, random_state=1, test_size=0.3)
model2 = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(trainX, trainY)
pred2 = model2.predict(testX)
print('예측값 : ', pred2[:5])
print('정확도 : ', sklearn.metrics.accuracy_score(testY, pred2))    #0.9502564102564103
