"""
Validation split 방식과 K-fold 방식의 차이

Validation split: 검증을 목적으로 임시 분할
K-fold: 평균을 내서 평가목적
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 데이터 가져오기
data = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/pima-indians-diabetes.data.csv', delimiter=',', dtype=np.float32)
print(data[:3])
x = data[:, :-1]
y = data[:, -1]
print(x[:3])
print(y[:3])


# 데이터 설계하기
def buildModel():
    model = Sequential([
        Input(shape=(8,)),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Validation split 방식 사용하기
modelVal = buildModel()
historyVal = modelVal.fit(x, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
valAcc = historyVal.history['val_accuracy'][-1]

# K-fold 방식 사용하기
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kfoldAccs = []

for train_idx, val_idx in kf.split(x):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    modelKfd = buildModel()
    modelKfd.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    yPred = modelKfd.predict(x_val)
    yPredLabel = (yPred > 0.5).astype(int)
    acc = accuracy_score(y_val, yPredLabel)
    kfoldAccs.append(acc)

# Validation split과 K-fold 비교 : 
print(f'[Validation_split] 마지막 검증 정확도 : {valAcc:.4f}')
print(f'[K-fold] 각 폴드의 정확도 : {np.round(kfoldAccs, 4)}')
print(f'[K-fold] 평균 정확도 : {np.mean(kfoldAccs):.4f}')