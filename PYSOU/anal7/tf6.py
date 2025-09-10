# 넘파이와 텐서플로우, 케라스 관련 모듈을 불러옴
import numpy as np                           # 수치 계산과 배열 처리를 위한 NumPy 모듈
import tensorflow as tf                      # 텐서플로우 라이브러리 (딥러닝 프레임워크)
from keras.models import Sequential          # 순차 모델(Sequential Model)을 사용하기 위한 클래스
from keras.layers import Dense, Activation, Input  # 신경망 레이어(Dense, Activation)와 입력 정의를 위한 Input 클래스
from keras.optimizers import SGD, RMSprop, Adam     # 다양한 최적화 알고리즘들 (SGD, RMSprop, Adam)

# XOR 문제에 사용할 입력(x)과 출력(y) 데이터를 정의
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])    # 입력 데이터 (4가지 조합)
y = np.array([[0], [1], [1], [0]])               # XOR 문제의 정답 (출력값)

# 신경망 모델 생성 (Sequential: 순차적으로 레이어를 쌓음)
model = Sequential()                             # 순차 모델 객체 생성

# 입력 레이어를 정의 (입력은 2개의 값으로 구성)
model.add(Input(shape=(2,)))                     # 입력 뉴런 수 2개, 입력의 형태 정의

# 은닉층(hidden layer) 추가 (뉴런 5개, 활성화 함수는 ReLU)
model.add(Dense(units=5, activation='relu'))     # 은닉층, 뉴런 수 5개, ReLU 활성화 함수 사용

# 출력층 추가 (출력 뉴런 1개, 활성화 함수는 sigmoid로 이진 분류)
model.add(Dense(units=1, activation='sigmoid'))  # 출력층, 뉴런 1개, sigmoid 활성화 함수

# 모델 구조 요약 출력
print(model.summary())                           # 모델의 전체 구조와 파라미터 수를 출력

"""
모델 구조 설명:
 - 첫 번째 Dense 레이어: 입력 뉴런 2개 + 편향(bias) 1개 => (2+1)*5 = 15개 파라미터
 - 두 번째 Dense 레이어: 은닉층 뉴런 5개 + 편향 1개 => (5+1)*1 = 6개 파라미터
"""

# 모델 컴파일 (손실 함수와 최적화 방법, 평가 지표 지정)
model.compile(loss='binary_crossentropy',        # 이진 분류 문제이므로 binary_crossentropy 사용
              optimizer=Adam(learning_rate=0.01),# Adam 최적화 알고리즘 사용, 학습률은 0.01
              metrics=['accuracy'])              # 정확도를 평가 지표로 사용

# 모델 학습 (fit 함수를 사용하여 입력 데이터와 정답을 기반으로 학습)
history = model.fit(x, y,                        # 학습 데이터 (입력 x와 출력 y)
                    epochs=100,                  # 전체 데이터셋을 100번 반복 학습
                    batch_size=1,                # 배치 크기 1 (한 번에 하나씩 학습)
                    verbose=0)                   # verbose=0이면 학습 과정 출력 생략

# 학습 후 모델 평가 (evaluate 함수를 사용하여 손실값과 정확도 측정)
loss_metrics = model.evaluate(x, y)              # 평가 데이터로 입력 x와 정답 y 사용
print('loss_metrics : ', loss_metrics)           # 평가 결과 출력 (손실값과 정확도)

# 학습된 모델을 사용하여 예측 수행
pred = (model.predict(x) > 0.5).astype('int32')  # 예측 결과가 0.5 초과면 1, 아니면 0으로 변환
print('예측 결과 : ', pred.ravel())              # 예측 결과를 1차원 배열로 출력

# 학습된 모델의 가중치(weights) 출력
print(model.weights)                             # 각 레이어의 가중치와 편향 출력

# 학습 중 손실 값(loss) 출력 (처음 10개만)
print(history.history['loss'][:10])              # 학습 손실(loss) 값의 변화 (처음 10 epoch)

# 학습 중 정확도(accuracy) 출력 (처음 10개만)
print(history.history['accuracy'][:10])          # 학습 정확도 값의 변화 (처음 10 epoch)

# 학습 과정을 시각화 (손실과 정확도 그래프)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], color='red', label='train loss')
plt.plot(history.history['accuracy'], label='train acc')
plt.xlabel('epochs')
plt.legend(loc='best')
plt.show()
plt.close()
