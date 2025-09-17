"""
iris 데이터 세트로 다항 분류 후 모델 성능 확인 : ROC 커브 까지.

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input

# 데이터 가져오기
iris = load_iris()
print(iris.keys())  #['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
x = iris.data
y = iris.target
print(x[:2])
print(y[:2])
print(set(y))

names = iris.target_names   # ['setosa' 'versicolor' 'virginica']
print(names)
feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(feature_names)

# 레이블에 대한 원 핫 인코딩 처리
onehot = OneHotEncoder(categories='auto')   # 케라스의 to_categorical, 넘파이의 np.eye(), 판다스의 pd.get_dummies()
print(y.shape)      # (150,)
y = onehot.fit_transform(y[:, np.newaxis]).toarray()
print(y.shape)      # (150, 3), 원핫처리가 되면서 3개의 칼럼으로 나뉨.
print(y[:2])

# 피쳐에 대한 표준화 실시
print(x[:2])        #[5.1 3.5 1.4 0.2], [4.9 3.  1.4 0.2]
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])  #[-0.90068117  1.01900435 -1.34022653 -1.3154443 ],[-1.14301691 -0.13197948 -1.34022653 -1.3154443 ]

# train_test_split 실시
xTrain, xTest, yTrain, yTest = train_test_split(x_scale, y, test_size=0.3, random_state=1)
n_features = xTrain.shape[1]
n_classes = yTrain.shape[1]
print(n_features, n_classes)    # 4, 3

# 신경망 모델 설계하기
def createModelFunction(input_dim, output_dim, out_nodes, n, model_name='model'):
    # print(input_dim, output_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name=model_name)
        model.add(Input(shape=(input_dim,)))
        for i in range(n):
            model.add(Dense(units=out_nodes, activation='relu'))
        model.add(Dense(units=output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model
    return create_model     #클로져

models = [createModelFunction(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1, 4)]
print(len(models))  # 3

for create_model in models:
    print()
    create_model().summary()

history_dict = {}
for create_model in models:
    print()
    model = create_model()
    print('모델명 : ', model.name)
    historys = model.fit(xTrain, yTrain, batch_size=8, epochs=50, verbose=0, validation_split=0.2)
    score = model.evaluate(xTest, yTest, verbose=0)
    print('test dataset loss : ', score[0])
    print('test dataset acc : ', score[1])

    history_dict[model.name] = [historys, model]

print(history_dict)

# 모델 학습률 시각화 해보기
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

for model_name in history_dict:
    print('h_d : ', history_dict[model_name][0].history['acc'])
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label = model_name)
    ax2.plot(val_loss, label = model_name)
    ax1.set_ylabel('val_acc')
    ax2.set_ylabel('val_loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()

plt.show()

# ROC Curve : 분류기에 대한 성능 평가 방법 중 하나.
from sklearn.metrics import roc_curve, auc

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
for model_name in history_dict:
    model = history_dict[model_name][1]
    y_pred = model.predict(xTest)
    fpr, tpr, _ = roc_curve(yTest.ravel(), y_pred.ravel())
    plt.plot(fpr, tpr, label='{}, auc value : {:.3f}'.format(model_name, auc(fpr, tpr)))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
plt.show()