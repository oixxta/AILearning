"""
세계 정치인들 얼굴 사진을 이용한 분류 모델 실습 : 

SVM 모델은 딥러닝을 제외한 모든 모델들 중 유일하게 이미지 분류가 가능함.
"""
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

faces = fetch_lfw_people(min_faces_per_person=60, color=False, resize=0.5)
#print(faces.DESCR)
print(faces.data)
print(faces.data.shape) #(1348, 2914)
print(faces.target)
print(faces.target_names)   #['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush' 'Gerhard Schroeder' 'Hugo Chavez' 'Junichiro Koizumi' 'Tony Blair']
print(faces.images.shape)   #(1348, 62, 47), 3차원 형태, 62 * 47 픽셀 크기의 그림 1348개가 있음.

print(faces.images[1])
"""
[[0.28627452 0.20784314 0.2535948  ... 0.28496733 0.3620915  0.30457518]
 [0.24836601 0.22745098 0.33594772 ... 0.27189544 0.34901962 0.30588236]
 [0.29281047 0.3006536  0.37908497 ... 0.25751635 0.33594772 0.32941177]
 ...
 [0.10718954 0.10326798 0.08104575 ... 0.95032674 0.9267974  0.90718955]
 [0.09411765 0.07843138 0.07189543 ... 0.96993464 0.951634   0.9359477 ]
 [0.07973856 0.05359477 0.06405229 ... 0.96993464 0.95032686 0.9346406 ]]
각 숫자들은 픽셀의 점 유무를 표시.
"""
# print(faces.target_names[faces.target[1]])
# plt.imshow(faces.images[1], cmap='bone')
# plt.show()
# plt.close()

"""
fig, ax = plt.subplots(3, 5)
#print(fig)          #Figure(640x480)
#print(ax.flat)      #
#print(len(ax.flat))

for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    
plt.show()
"""


# 주성분분석으로 이미지 차원을 축소해 분류작업 진행해보기
m_pca = PCA(n_components=150, whiten=True, random_state=0)
x_low = m_pca.fit_transform(faces.data)
print('x_low : ', x_low, ' ', x_low.shape)      # (1348, 150), 2914에서 150개로 줄어듬.

m_svc = SVC(C=1)
model = make_pipeline(m_pca, m_svc)     #PCA와 분류기를 하나로 묶어서 순차적으로 실시
print(model)    #Pipeline(steps=[('pca', PCA(n_components=150, random_state=0, whiten=True)),('svc', SVC(C=1))])

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(faces.data, faces.target, test_size=0.3, random_state=1)
model.fit(xTrain, yTrain)
pred = model.predict(xTest)
print('예측값 : ', pred[:10])   #[1 4 1 3 3 3 7 3 3 3]
print('실제값 : ', yTest[:10])  #[1 4 1 5 3 2 7 3 1 3]
from sklearn.metrics import classification_report
print(classification_report(yTest, pred, target_names=faces.target_names))
"""
                   precision    recall  f1-score   support

     Ariel Sharon       1.00      0.47      0.64        17
     Colin Powell       0.85      0.86      0.85        65
  Donald Rumsfeld       1.00      0.38      0.55        37
    George W Bush       0.66      0.98      0.79       163
Gerhard Schroeder       0.96      0.65      0.77        37
      Hugo Chavez       1.00      0.36      0.53        22
Junichiro Koizumi       1.00      0.57      0.73        14
       Tony Blair       0.97      0.62      0.76        50

         accuracy                           0.76       405
        macro avg       0.93      0.61      0.70       405
     weighted avg       0.83      0.76      0.75       405
"""
from sklearn.metrics import confusion_matrix, accuracy_score
mat = confusion_matrix(yTest, pred)
print('컨퓨전 메트릭스 : \n', mat)
"""
[[  8   1   0   8   0   0   0   0]
 [  0  56   0   9   0   0   0   0]
 [  0   1  14  21   0   0   0   1]
 [  0   3   0 160   0   0   0   0]
 [  0   1   0  12  24   0   0   0]
 [  0   2   0  12   0   8   0   0]
 [  0   1   0   5   0   0   8   0]
 [  0   1   0  17   1   0   0  31]]
"""
print('정확도 점수 : \n', accuracy_score(yTest, pred))  #  0.762962962962963, 정확도 76%


#분류 결과를 시각화 해보기
#xTest[0] 한 개만 확인
#print(xTest[0], ' ', xTest[0].shape)
#print(xTest[0].reshape(62, 47))         #이미지 출력 시 이처럼 1차원을 2차원으로 변환시켜줘야 함!, x축과 y축이 필요하기 때문.
#plt.subplots(1, 1)
#plt.imshow(xTest[0].reshape(62, 47), cmap='bone')
#plt.show()

#24개의 분류결과를 4행 6열로 보기
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(xTest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[pred[i]].split()[-1], color='black' if pred[i] == yTest[i] else 'red')     
    #라스트 네임만 사용, 정답 시 검은색 이름, 오답 시 붉은색 이름 출력
fig.suptitle('pred result', size=14)
plt.show()

#오차행렬 시각화
import seaborn as sns
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true(real) label')
plt.ylabel('pred label')
plt.show()