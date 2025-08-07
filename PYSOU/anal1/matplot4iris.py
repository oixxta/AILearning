#아이리스(붓꽃) 데이터 셋(iris dataset) : 꽃밭침과 꽃잎의 너비와 길이로 꽃의 종류를 
# 3가지로 구문해 놓은 데이터

# 각 그룹당 50개, 총 150개의 데이터

import pandas as pd
import matplotlib.pyplot as plt

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv')
print(iris_data.info())
print(iris_data.head(3))
print(iris_data.tail(3))

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length']) #산포도 그리기
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.show()

print()
print(iris_data['Species'].unique())
print(set(iris_data['Species']))

cols = []
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa': choice = 1
    elif s == 'versicolor' : choice = 2
    elif s == 'virginica' : choice = 3
    cols.append(choice)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('types of flowers')
plt.show()

#데이터 분포곡선과 산점도 그래프 같이 그리기
iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Length']
print(iris_col)
from pandas.plotting import scatter_matrix   # pandas의 시각화 기능 활용
scatter_matrix(iris_col, diagonal='kde')
plt.show()

import seaborn as sns
sns.pairplot(iris_data, hue='Species', height=2)    #Seaborn 시각화 기능 사용
plt.show()

# rug plot
x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

#kurnel density (커널 밀도)
sns.kdeplot(x)
plt.show()
