"""
zoo 데이터 세트로 다항분류 모델 만들기 연습해보기
"""
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
#from keras.utils import to_categorical  #다항분류때 중요함, 원핫 인코딩을 지원함.
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#데이터 가져오기
datas = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/zoo.csv')
print(datas.head(3))
print(datas.columns)
#['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'brethes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']
print(datas.shape)   #(101, 17)


#학습데이터 나누기
xData = datas.iloc[:, :-1].astype('float32').values
yData = datas.iloc[:, -1].astype('int32').values
print(xData[:2], xData.shape)
"""
[[1. 0. 0. 1. 0. 0. 1. 1. 1. 1. 0. 0. 4. 0. 0. 1.]
 [1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 0. 0. 4. 1. 0. 1.]] (101, 16)
"""
print(yData[:2], yData.shape)
"""
[0 0] (101,)
"""
nb_classes = len(set(yData))
print('classes 범주 : ', nb_classes)    # classes 범주 :  7

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42, stratify=yData)

#모델 정의하기 : Sequential
model = Sequential([
    Input(shape=(xData.shape[1])),
    Dense(units=64, activation='relu'),
    Dropout(0.3),
    Dense(units=32, activation='relu'),
    Dense(units=7, activation='softmax'),
])
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   #sparse_categorical_crossentropy : 알아서 원핫 인코딩을 해줌.
#Callback 사용해보기
earlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)   #restore_best_weights=True : 학습이 끝나고 난 다음 가장 좋은 val_loss 값을 기록한 epoch의 가중치.
checkpoint = ModelCheckpoint('best_zoom_model.keras', monitor='val_loss', save_best_only=True)
history = model.fit(xTrain, yTrain, epochs=1000, validation_split=0.2, callbacks=[earlyStop, checkpoint], verbose=1)


#모델 평가하기
loss, acc = model.evaluate(xTest, yTest, verbose=0)
print(f'최종 평가 : Loss : {loss:.4f}, accuracy : {acc:.4f}')

#학습에 대한 곡선 시각화 해보기
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], '--', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], '--', label='val accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.close()

#컨퓨전 메트릭스(혼돈행렬)와 리포트 보기
yPred = np.argmax(model.predict(xTest), axis=1)
print(yPred)
print('Report : ', classification_report(yTest, yPred))

cm = confusion_matrix(yTest, yPred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#저장한 모델('best_zoom_model.keras')을 불러와서 예측 해보기
from keras.models import load_model
bestModel = load_model('best_zoom_model.keras')
loss, acc = bestModel.evaluate(xTest, yTest, verbose=0)
print(f'최종 평가 : Loss : {loss:.4f}, accuracy : {acc:.4f}')

#새로운 데이터로 새 데이터를 분류해오기
newData = np.array([[1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 52., 0., 0., 1.]])
probs = bestModel.predict(newData)
print(probs)
pred_class = np.argmax(probs)
print('예측 결과 : ', pred_class)
