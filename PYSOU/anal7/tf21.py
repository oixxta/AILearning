"""
다항분류 : 출력값이 softmax 함수로 인해 확률값이 여러개의 확률값으로 출력.
이 때, 확률값이 가장 큰 인덱스를 분류의 결과로 얻음.

softmax 함수를 한번 코드화 시켜보기.

import numpy as np

def softmaxFunc(a):
    c = np.max(a)       #숫자값이 너무 커질때 오버플로우 방지용.
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1.0, 1.2, 1.5])
result = softmaxFunc(a)
print(result)   #[0.25838965 0.31559783 0.42601251], 새 개의 숫자는 확률값임.
#softmax 함수는 세 확률 중 가장 확률이 큰 것이 맞는것으로 간주하는 것으로 분류를 함.
"""

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical  #다항분류때 중요함, 원핫 인코딩을 지원함.
import matplotlib.pyplot as plt
import numpy as np

np.random.seed()
#데이터 준비
xData = np.random.random((1000, 12))    # 시험점수라고 가정.
yData = np.random.randint(5, size=(1000, 1))
print(xData[:2])    # feature
print(yData[:2])    # label     소프트맥스 처리엔 정수를 5가지 형태로 출력될 수 있도록 모양 변경(원핫 인코딩) 필요
yData = to_categorical(yData, num_classes=5)
print(yData[:2])
"""
원핫 인코딩을 거치면서의 변화 : 
[4] > [0. 0. 0. 0. 1.]
[3] > [0. 0. 0. 1. 0.]
"""
#print([np.argmax(i) for i in yData[:2]]) #원핫 인코딩을 원래 값으로 되돌리는 방법 : 아규먼트 중 가장 큰값의 인덱스을 반환함.


#모델 만들기
model = Sequential()
model.add(Input(shape=(12,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #다항일 때는 카테코리컬 크로스엔트로피를 사용함.
print('learning rate of adam : ', model.optimizer.learning_rate.numpy())    #0.001

history = model.fit(xData, yData, epochs=1000, batch_size=32, verbose=0) #yData는 원핫 인코딩이 되어있는 상태.
model_eval = model.evaluate(xData, yData)
print('모델 평가 결과 : ', model_eval)


#모델 평가결과 시각화 해보기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
ax1.plot(history.history['loss'])
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')

ax2.plot(history.history['accuracy'])
ax2.set_title('Accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')

plt.show()
plt.close()


#분류 예측 결과 보기
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
print('예측값 : ', model.predict(xData[:5]))
print('예측값 : ', np.argmax(model.predict(xData[:5]), axis=1))
print('실제값 : ', yData[:5])
print('실제값 : ', [int(i) for i in np.argmax(yData[:5], axis=1)])

#새로운 값으로 예측하기
xNew = np.random.random([1, 12])
print(xNew)
newPred = model.predict(xNew)
print('분류 결과 : ', newPred, ' 모두 더하면 : ', np.sum(newPred))
print('분류 결과 : ', np.argmax(newPred))


#가정 : 레이블에 해당하는 과목명 출력해보기
classes = np.array(['국어', '영어', '수학', '과학', '체육'])
print('예측값 : ', classes[np.argmax(newPred, axis=1)])


