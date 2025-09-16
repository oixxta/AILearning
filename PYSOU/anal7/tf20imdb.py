"""
영화 리뷰 : IMDB 데이터 세트를 10000개만 가져오기.(전체는 50000개)
이진분류 : good or bad(극정 or 부정)
"""

#원본 데이터
from keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)
#!ls - al /root/.keras/datasets
print(train_data)   # 영화 인덱스 번호
print(train_label)  # 1 혹은 0
print(train_data[0], len(train_data[0]))
#참고 : 리뷰데이터 하나를 원래 영어 단어로 보기.
word_index = imdb.get_word_index()
print(word_index)
print(word_index.items())
revers_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(revers_word_index)
for i, (k, v) in enumerate(sorted(revers_word_index.items(), key=lambda x:x[0])):
    if i >= 10:
        break
    print(k, ':', v)
decord_review = ' '.join([revers_word_index.get(i) for i in train_data[10]])
print(decord_review)


#데이터 준비
import numpy as np
def vector_seq(sequences, dim = 10000):
    results = np.zeros((len(sequences), dim))   #크기가 (len(sequences), dim)이고, 모든값이 0인 행렬
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

xTrain = vector_seq(train_data) #train_data (list -> Vector로 변환)
xTest = vector_seq(test_data)
#print(xTrain, ' ', xTrain.shape)
"""
[[0. 1. 1. ... 0. 0. 0.]
 [0. 1. 1. ... 0. 0. 0.]
 [0. 1. 1. ... 0. 0. 0.]
 ...
 [0. 1. 1. ... 0. 0. 0.]
 [0. 1. 1. ... 0. 0. 0.]
 [0. 1. 1. ... 0. 0. 0.]]   (25000, 10000)
"""
yTrain = train_label.astype('float32')
yTest = test_label.astype('float32')
#print(yTrain, ' ', yTrain.shape)
"""
[1. 0. 0. ... 0. 1. 0.]   (25000,)
"""

#모델 작성하기(신경망 구성)
from keras import models, layers, regularizers
model = models.Sequential()
model.add(layers.Input(shape=(10000,)))
model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])    #optimizer의 종류 4가지를 외울것
print(model.summary())
#validation data 준비하기
xVal = xTrain[:10000]               #10000개를 직접 자름.
partialXTrain = xTrain[10000:]      #남은 트레인 데이터 15000개
yVal = yTrain[:10000]
pratialYTrain = yTrain[10000:]
#모델 학습
history = model.fit(partialXTrain, pratialYTrain, epochs=30, batch_size=512, validation_data=(xVal, yVal), verbose=2)
#history = model.fit(xTrain, yTrain, epochs=30, batch_size=512, validation_split=0.2, verbose=2)    #validation 데이타를 안쓸 경우 이렇게 할 수 있음.


#훈련 검증하기
#손실과 정확도에 대한 시각화 하기
import matplotlib.pyplot as plt
history_dict = history.history
print(history_dict.keys())  #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
loss = history_dict['loss']
valLoss = history_dict['val_loss']
acc = history_dict['acc']
valAcc = history_dict['val_acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, valLoss, 'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()   #그래프 초기화, plt.close()랑은 다른개념임.

plt.plot(epochs, acc, 'bo', label='train loss')
plt.plot(epochs, valAcc, 'r', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.close() #확인 결과, 과적합이 우려됨.
#과적합 방지를 위한 방법 : 
# 1. 모델의 파라미터 조절, 배치사이즈 조절. 
# 2. 히든레이어에 가중치 규제 실시, 
# 3. 레이어 사이에 드랍아웃 추가. 
# 4. 훈련데이터의 비율을 더 늘림(만약 안했다면, train_test_split 실시) 
# 5. K-Fold 실시

#모델로 예측해보기
yPred = model.predict(xTest[:5])
print('예측값 : ', np.where(yPred >= 0.5, 1, 0).ravel())    #예측값 :  [0  1  0  1  1 ]
print('실제값 : ', yTest[:5])                               #실제값 :  [0. 1. 1. 0. 1.]