"""


"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import tensorflow as tf

# 데이터 가져오기 및 전처리
url = 'https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true'
trainDf = pd.read_excel(url, sheet_name='train')
testDf = pd.read_excel(url, sheet_name='test')
print(trainDf.head(2))
"""
     가격    년식   종류    연비   마력    토크   연료  하이브리드   배기량    중량 변속기
0  1885  2015  준중형  11.8  172  21.0  가솔린      0  1999  1300  자동
1  2190  2015  준중형  12.3  204  27.0  가솔린      0  1591  1300  자동
"""
print(testDf.head(2))

xTrain = trainDf.drop(['가격'], axis=1) #가격을 제외한 나머지 feature로
xTest = testDf.drop(['가격'], axis=1)
yTrain = trainDf[['가격']]              #가격을 label로
yTest = testDf[['가격']]

print(xTrain.head(2))
print(xTrain.columns)       #Index(['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기'], dtype='object')
print(xTrain.shape)         #(71, 10)

print(set(xTrain.종류))     #{'소형', '중형', '준중형', '대형'}
print(set(xTrain.연료))     #{'LPG', '가솔린', '디젤'}
print(set(xTrain.변속기))   #{'자동', '수동'}

#종류, 연료, 변속기 이상 3개의 칼럼에 대해서는 labelEncoder(), OneHotEncoder() 등을 적용.
transformer = make_column_transformer((OneHotEncoder(), ['종류', '연료', '변속기']), remainder='passthrough')
# remainder : 기본값은 'drop', 'passthrough'를 지정할 경우, 열이 객체변수(transformer)로 전달됨.
transformer.fit(xTrain)
xTrain = transformer.transform(xTrain)  # 3개의 칼럼을 포함해, 모든 칼럼이 표준화가 되었음.
xTest = transformer.transform(xTest)
print(xTest[:2], xTrain.shape)
"""
[[1.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00 0.000e+00
  1.000e+00 0.000e+00 2.015e+03 6.800e+00 1.590e+02 2.300e+01 0.000e+00
  2.359e+03 1.935e+03]
 [0.000e+00 1.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 0.000e+00
  0.000e+00 1.000e+00 2.012e+03 1.330e+01 1.080e+02 1.390e+01 0.000e+00
  1.396e+03 1.035e+03]] (71, 16)
"""
print(yTrain[:2], yTrain.shape)
"""
0  1885
1  2190 (71, 1)
"""