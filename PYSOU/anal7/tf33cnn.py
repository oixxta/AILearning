"""
MNIST 데이터 세트로 CNN 모델 작성해보기

Model Subclassing API 사용
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 가져오기 
(x_Train, y_Train), (x_Test, y_Test) = tf.keras.datasets.mnist.load_data()

# 구조 변경(차원)
# CNN은 witdth, height, 행, channel 정보가 담겨야 하기 때문에 4개의 입력값이 필요함.
print(x_Train.shape)    # (60000, 28, 28)
x_Train = x_Train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_Test = x_Test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_Train.shape)    # (60000, 28, 28, 1)
#print(x_Train)

# 모델 정의하기 : Model Subclassing API
#Model Subclassing API(모델, 레이어, 함수 : 손실, 활성화)를 모델 저장 시 자동으로 직렬화 시스템에 등록해 주는 역할을 함.
@tf.keras.utils.register_keras_serializable(package='custom')   # 'losses', 'activation'
class MyMnistCnn(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D((2, 2))
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(rate=0.3)
        self.d2 = tf.keras.layers.Dense(units=32, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(rate=0.2)
        self.outputs = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):  #오버라이딩 메서드
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flat(x)
        x = self.d1(x)
        x = self.drop1(x, training = True)  #드랍아웃은 트레이닝이 필요함
        x = self.d2(x)
        x = self.drop2(x, training = True)
        return self.outputs(x)
    
    
model = MyMnistCnn()
model.build(input_shape=(None, 28, 28, 1))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(x_Train, y_Train, epochs=100, batch_size=128, validation_split=0.1, callbacks=[es], verbose=2)

# 모델 평가
train_loss, train_acc = model.evaluate(x_Train, y_Train, verbose=0)
test_loss, test_acc = model.evaluate(x_Test, y_Test, verbose=0)
print(f'train_loss : {train_loss:.4f}, train_acc : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f}, test_acc : {test_acc:.4f}')