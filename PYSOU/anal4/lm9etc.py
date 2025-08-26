import joblib
import pandas as pd
ourModel = joblib.load('myModel.model')
newDf = pd.DataFrame({'Income':[44, 44, 44], 'Advertising':[6, 3, 11], 'Price':[105, 88, 77], 'Age':[33, 55, 22]})
new_pred = ourModel.predict(newDf)
print('Sales 예측 결과 :\n', new_pred)

"""
Sales 예측 결과 :
0     8.761168
1     8.303319
2    11.501090
"""