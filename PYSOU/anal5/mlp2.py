"""
MLP 실습 : 종양 데이터


"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']

xTrain, xTest, yTrain, yTest = train_test_split(x, y)
sc = StandardScaler()
sc.fit(xTrain)
sc.fit(xTest)
xTrain = sc.transform(xTrain)
xTest = sc.transform(xTest)

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), solver='adam', learning_rate_init=0.1, verbose=1)  #verbose : 학습 과정 지켜봄.
mlp.fit(xTrain, yTrain)
pred = mlp.predict(xTest)
print('예측값 : ', pred[:5])        #예측값 :  [0 1 0 1 0]
print('실제값 : ', yTest[:5])       #실제값 :  [0 1 0 1 0]
print('정확도 : ', accuracy_score(yTest, pred)) #0.986013986013986


