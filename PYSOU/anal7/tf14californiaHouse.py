"""
캘리포니아 주택 가격 데이터로 함수형 유연한 모델 생성

"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
import matplotlib.pylab as plt


housing = fetch_california_housing()
print(housing.keys())
print(housing.data[:3], type(housing.data))
print(housing.target[:3], type(housing.target)) #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(housing.feature_names)    #['MedHouseVal']
print(housing.target_names)
print(housing.data.shape)       #(20640, 8)

xTrain, xTest, yTrain, yTest = train_test_split(housing.data, housing.target, test_size=0.2, random_state=12)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape) #(16512, 8) (4128, 8) (16512,) (4128,)


model = Sequential()
model.add(Input(1))