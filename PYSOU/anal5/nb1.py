"""
나이브베이즈

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(df.head(2))
"""
         Date  MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp RainToday RainTomorrow
0  2016-11-01      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6        No          Yes
1  2016-11-02     14.0     26.9       3.6       9.7         17        36    1008.4      3  25.7       Yes          Yes
"""
print(df.info())
"""
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   Date          366 non-null    object
 1   MinTemp       366 non-null    float64
 2   MaxTemp       366 non-null    float64
 3   Rainfall      366 non-null    float64
 4   Sunshine      363 non-null    float64
 5   WindSpeed     366 non-null    int64
 6   Humidity      366 non-null    int64
 7   Pressure      366 non-null    float64
 8   Cloud         366 non-null    int64
 9   Temp          366 non-null    float64
 10  RainToday     366 non-null    object
 11  RainTomorrow  366 non-null    object
"""
x = df[['MinTemp', 'MaxTemp', 'Rainfall']]
label = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(x[:3])
print(label[:3])

trainX, testX, trainY, testY = train_test_split(x, label, test_size=0.3, random_state=0)
gModel = GaussianNB()
gModel.fit(trainX, trainY)
pred = gModel.predict(testX)
print("예측값 : ", pred[:10])
print("실제값 : ", testY[:10].values)
acc = sum(testY == pred) / len(pred)
print('정확도 1 : ', acc)                               #0.7636363636363637
print('정확도 2 : ', accuracy_score(testY, pred))       #0.7636363636363637

