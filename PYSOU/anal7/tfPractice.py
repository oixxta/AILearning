"""
[문항13] MNIST 손글씨 데이터셋을 이용하여 간단한 CNN 분류 모델을 설계하고 학습하고 정확도도 출력하는 코드를 작성하시오.

조건은 다음과 같다.
- 입력 데이터는 (28, 28, 1) 크기의 흑백 이미지이다.
- 첫 번째 합성곱 층(Conv2D)은 필터 수 32개, 커널 크기 (3,3), 활성화 함수 relu를 사용한다.
- 합성곱 층 뒤에는 (2,2) 크기의 MaxPooling2D를 적용한다.
- Flatten 층을 사용하여 Dense 층과 연결한다.
- 출력층은 클래스 개수(10개)에 맞춰 Dense(10, activation="softmax")로 구성한다.
- label은 원핫 처리 하지 않음
- Optimizer는 'adam'을 사용한다.
- 학습 횟수는 3으로 수행한다.
"""

import tensorflow as tf

(x_Train, y_Train), (x_Test, y_Test) = tf.keras.datasets.mnist.load_data()
x_Train = x_Train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_Test = x_Test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

myModel = tf.keras.models.Sequential()

myModel.add(tf.keras.layers.Input(shape=(28, 28, 1)))
myModel.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
myModel.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
myModel.add(tf.keras.layers.Flatten())
myModel.add(tf.keras.layers.Dense(units=64, activation='relu'))
myModel.add(tf.keras.layers.Dense(units=32, activation='relu'))
myModel.add(tf.keras.layers.Dense(units=10, activation='softmax'))

myModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = myModel.fit(x_Train, y_Train, epochs=3, batch_size=128, validation_split=0.1, verbose=2)

train_loss, train_acc = myModel.evaluate(x_Train, y_Train, verbose=0)
test_loss, test_acc = myModel.evaluate(x_Test, y_Test, verbose=0)
print(f'train_loss : {train_loss:.4f}, train_acc : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f}, test_acc : {test_acc:.4f}')
