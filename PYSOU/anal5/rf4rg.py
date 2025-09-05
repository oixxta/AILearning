"""
RandomForestRegressor : 정량적 예측 모델 작성에 사용

데이터 : 캘리포니아 하우징 데이터 세트 사용
"""
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = fetch_california_housing(as_frame=True)  #사이키런에서 받은 파일을 바로 데이터프레임화
print(data.data[:2])
"""
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
0  8.3252      41.0  6.984127    1.02381       322.0  2.555556     37.88    -122.23
1  8.3014      21.0  6.238137    0.97188      2401.0  2.109842     37.86    -122.22
"""
print(data.target[:2])
"""
0    4.526
1    3.585
"""
print(data.feature_names)
"""
['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
"""
df = data.frame     #데이터 프레임화, as_frame=True 덕분에 가능!
print(df.head(2))
"""
   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
0  8.3252      41.0  6.984127    1.02381       322.0  2.555556     37.88    -122.23        4.526
1  8.3014      21.0  6.238137    0.97188      2401.0  2.109842     37.86    -122.22        3.585
"""

#feature / label로 분리하기
xData = df.drop('MedHouseVal', axis = 1)
yData = df['MedHouseVal']

#테스트/학습 데이터 나누기
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3, random_state=42)

#랜덤포레스트 모델 만들기
rfModel = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)   #n_estimators : 디시전 트리 갯수, n_jobs : 사용 가능한 CPU 코어 수, -1은 모두 사용.
rfModel.fit(xTrain, yTrain)

#모델 테스트
y_pred = rfModel.predict(xTest)
print('MSE : ', mean_squared_error(yTest, y_pred))  #MSE :  0.25355774432775086
print('R² : ', r2_score(yTest, y_pred))             #R² :  0.8068190504669066    , 여러 독립변수들이 종속변수와의 연관이 80%에 가까움.

importance = rfModel.feature_importances_
indices = np.argsort(importance)[::-1]          #내림차순 정렬

print('독립변수 중요도 순위 표')
ranking = pd.DataFrame({
    'Feature' : xData.columns[indices],
    'Importance' : importance[indices]
})
print(ranking)
"""
독립변수 중요도 순위 표
      Feature  Importance
0      MedInc    0.525400
1    AveOccup    0.138819
2   Longitude    0.086695
3    Latitude    0.086512
4    HouseAge    0.054694
5    AveRooms    0.045933
6  Population    0.032089
7   AveBedrms    0.029859
"""

# 간단한 튜닝으로 최적의 파라미터 찾기
# GridSearchCV : 정확하게 최적값 찾기에 적당. 하지만, 파라미터가 많으면 계산량 폭발적 증가.
# RandomizedSearchCV : 연속적 값 처리 가능, 하지만 최적 조합 못 찾을 수도 있음.
from sklearn.model_selection import RandomizedSearchCV
paramList = {
    'n_estimators' : [200, 400, 600],
    'max_depth' : [None, 10, 20, 30],
    'min_samples_leaf' : [1, 2, 4],                         # 리프노드 최수 샘플 수
    'min_samples_split' : [2, 4, 10],                       # 노드 분할 최소 샘플 수
    'max_features' : [None, 'sqrt', 'log2', 1.0, 0.8, 0.6]  # 최대 특성 수  log2(features)
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),     #기준모델
    param_distributions = paramList,            #파라미터 후보들
    n_iter=10,                                  #랜덤하게 10번 조합을 뽑아 평가하기
    scoring='r2',                               
    cv=3,
    random_state=42
)
search.fit(xTrain, yTrain)      # 탐색 수행(학습) 시작
print('beat params : ', search.best_params_)     #{'n_estimators': 600, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 0.6, 'max_depth': 30}
bestModel = search.best_estimator_              #최적 모델 추출
print('best cvr² (교차검증 평균 결정계수) : ', search.best_score_)  #best cvr² (교차검증 평균 결정계수) :  0.8038233023367537
print('bestModel 결정계수 : ', r2_score(yTest, bestModel.predict(xTest)))   #bestModel 결정계수 :  0.8133875957138825
