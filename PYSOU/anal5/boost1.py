"""
부스팅 : 
앙상블 기법 중 부스팅. 하나의 모델을 만들어 그 결과로 다른 모델을 만들어 나가는데, 샘플 데이터의 일부를 갱신하며 순차적으로
좀 더 강건한 모델을 생성하는 방법.
가중치를 활용하여 약분류기를 강분류기로 만드는 방법
단점으로는 너무 성능이 좋아서 오버피팅에 빠질 위험이 있음.

분류모델 만들기 : 유방암(brest_cancer) 데이터 세트로 분류모델을 만들기.

pip install xgboost  
pip install lightgbm 설치 필요
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from lightgbm import LGBMClassifier     # xgboost보다 성능 우수하나, 데이터 양이 적으면 과적합이 발생함!
import lightgbm as lgb

data = load_breast_cancer()
xData = pd.DataFrame(data.data, columns=data.feature_names)
yData = data.target
print(xData.shape)          #(569, 30)

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=12, stratify=yData)

#모델 만들기 : xgb 모델과 lgb 모델
xgbClf = xgb.XGBClassifier(
    booster = 'gbtree',          #gbtree:DecisionTree, gblinear:DecisionTree X
    maxDepth = 6,
    n_estimators = 500,
    eval_metric = 'logloss',        #logloss, error, rmse, ...
    random_state = 42
)

lgbClf = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)  #verbose : 학습 도중 로그값 출력 여부, -1로 하면 미출력, 1로 하면 출력

xgbClf.fit(xTrain, yTrain)
lgbClf.fit(xTrain, yTrain)

#예측 평가
predXgb = xgbClf.predict(xTest)
predLgb = lgbClf.predict(xTest)
print('XGBoost acc : ', accuracy_score(yTest, predXgb))     #XGBoost acc :  0.9649122807017544
print('LightGBM acc : ', accuracy_score(yTest, predLgb))    #LightGBM acc :  0.9912280701754386