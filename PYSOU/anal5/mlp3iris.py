
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import ListedColormap
import pickle
from sklearn.linear_model import LogisticRegression     #다중 클래스(종속변수, label, y, class) 지원
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

iris = datasets.load_iris()
#print(iris['data'])
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))
#[[1.         0.96286543]
# [0.96286543 1.        ]]
x = iris.data[:, [2, 3]]    # petal.length, petal.width만 작업에 이용
y = iris.target             # vector
print(x[:3])
#[[1.4 0.2]
# [1.4 0.2]
# [1.3 0.2]]
print(y[:3], set(y))
# [0 0 0]

# 학습데이터/연습데이터 분리(7:3 비율로)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(105, 2) (45, 2) (105,) (45,)

"""
# Scaling (독립변수의 데이터 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우 및 언더플로우 방지 효과가 있음.)
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)       #종속변수는 스케일링 하지 않음!
print(x_train[:3])

# Scaling 원복 (원래값으로 되돌림)
inver_x_train = sc.inverse_transform(x_train)
inver_x_test = sc.inverse_transform(x_test)

# Scaling은 원본으로 해도 잘 나오면 할 필요 없음!
"""

# 분류 모델을 생성
#model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0, min_samples_leaf=5)
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(30, 30, 30), solver='adam', learning_rate_init=0.1, max_iter=1000, verbose=1)



# C속성 : L2 규제 - 모델에 패널티 적용(Tuning parameter 중 하나.), 숫자 값을 조정해가며 분류 정확도 확인. 
# 값이 작을수록 더 강한 정규화 규제를 가함.
# vervose 속성 : 학습 과정을 보여지게 할지 안할지 결정. (기본값 0, 안보여줌.)
print(model)    #LogisticRegression(C=0.1, random_state=0), 기본값들은 무시함.
model.fit(x_train, y_train)     #지도학습, Supervised learning, 독립변수와 종속변수가 주어짐.

#분류 예측 - 모델 성능 파악용.
y_pred = model.predict(x_test)
print("예측값 : ", y_pred)  #[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2 1 1 2 0 2 0 0]
print("실제값 : ", y_test)  #[2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 1 1 1 2 0 2 0 0]
#1개 제외 전부 일치.
print('총 갯수 : ',len(y_test) ,'오류 수 : ', (y_test != y_pred).sum())
#총 갯수 :  45 오류 수 :  1

print('분류정확도 확인 1 :')
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류정확도 확인 2 :')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)
#print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류정확도 확인 3 :')
print('test : ', model.score(x_test, y_test))    #0.9777777777777777
print('train : ', model.score(x_train, y_train)) #0.9428571428571428
#두 개의 값 차이가 크면 과적합 의심!


#모델 외부파일로 저장하기
pickle.dump(model, open('logiModel.sav', 'wb'))  #모델 외부파일로 저장
del model   #코드 안의 model 지우기
read_model = pickle.load(open('logiModel.sav', 'rb'))  #외부파일 모델을 불러와서 read_model에 지정


#새로운 값 예측하기 : petal.length, petal.width만 참여
print(x_test[:3])
newData = np.array([[5.1, 1.1],[1.1, 1.1],[6.1, 7.1]])
# 참고 : 만약, 표준화된 데이터로 모델을 생성했다면, 
# sc.fit(newData); newData = sc.transform(newData)
# 위 과정을 거친 후에 아래 과정을 해야 함.
newPred = read_model.predict(newData)   # 내부적으로 softmax가 출력한 값을 argmax로 처리한 결과임.
print('예측 결과 : ', newPred)  #[2 0 2]
#print(read_model.predict_proba(newData))   # softmax가 출력한 값, predict()는 그 중 제일 큰것으로 반환함.
#[[3.05859126e-02 4.80004878e-01 4.89409209e-01],  2번 인덱스가 제일 커서 2로 예측됨
# [8.86247468e-01 1.10956949e-01 2.79558303e-03],  0번 인덱스가 제일 커서 0으로 예측됨
# [1.40841977e-05 4.79092000e-03 9.95194996e-01]], 2번 인덱스가 제일 커서 2로 예측됨


#시각화 하기
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx = range(100, 150), title='scikit-learn 제공')
