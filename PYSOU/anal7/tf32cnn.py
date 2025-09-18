"""
MNIST 데이터 세트로 CNN 모델을 작성해보기

Functional API 사용
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_Train, y_Train), (x_Test, y_Test) = tf.keras.datasets.mnist.load_data()

# 구조 변경(차원)
# CNN은 witdth, height, 행, channel 정보가 담겨야 하기 때문에 4개의 입력값이 필요함.
print(x_Train.shape)    # (60000, 28, 28)
x_Train = x_Train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_Test = x_Test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_Train.shape)    # (60000, 28, 28, 1)
#print(x_Train)

#모델 정의하기 : Functional API
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
"""    
# 방법 1 : 
x1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
x2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x1)
x3 = tf.keras.layers.Dropout(rate=0.2)(x2)

x4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x3)
x5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x4)

x6 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x5)
x7 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x6)

x8 = tf.keras.layers.Flatten()(x7)  #FCLayer (Fully-Connected Layer, 차원을 1차원으로 감소시킴.)
x9 = tf.keras.layers.Dense(units=64, activation='relu')(x8)
x10 = tf.keras.layers.Dropout(rate=0.3)(x9)
x11 = tf.keras.layers.Dense(units=32, activation='relu')(x10)
x12 = tf.keras.layers.Dropout(rate=0.3)(x11)
outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x12)
"""
# 방법 2 - BatchNormalization : Conv/Dense 뒤에 배치 - 학습을 안정화 시킴(수련 가속화).
# use_bias = False : Conv/Dense의 bias 제거
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)

x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)

x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(rate=0.25)(x)

x = tf.keras.layers.Flatten()(x)  #FCLayer (Fully-Connected Layer, 차원을 1차원으로 감소시킴.)
x = tf.keras.layers.Dense(units=32)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(rate=0.3)(x)

x = tf.keras.layers.Dense(units=32)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)


model = tf.keras.Model(inputs = inputs, outputs = outputs, name='MNIST_cnn_func')
print(model.summary())
"""
Model: "MNIST_cnn_func"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0

 conv2d (Conv2D)             (None, 28, 28, 16)        160

 max_pooling2d (MaxPooling2D  (None, 14, 14, 16)       0
 )

 dropout (Dropout)           (None, 14, 14, 16)        0

 conv2d_1 (Conv2D)           (None, 14, 14, 16)        2320

 max_pooling2d_1 (MaxPooling  (None, 7, 7, 16)         0
 2D)

 conv2d_2 (Conv2D)           (None, 7, 7, 16)          2320

 max_pooling2d_2 (MaxPooling  (None, 3, 3, 16)         0
 2D)

 flatten (Flatten)           (None, 144)               0

 dense (Dense)               (None, 64)                9280

 dropout_1 (Dropout)         (None, 64)                0

 dense_1 (Dense)             (None, 32)                2080

 dropout_2 (Dropout)         (None, 32)                0

 dense_2 (Dense)             (None, 10)                330

=================================================================
"""

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(x_Train, y_Train, epochs=100, batch_size=128, validation_split=0.1, callbacks=[es], verbose=2)

#모델 평가
train_loss, train_acc = model.evaluate(x_Train, y_Train, verbose=0)
test_loss, test_acc = model.evaluate(x_Test, y_Test, verbose=0)
print(f'train_loss : {train_loss:.4f}, train_acc : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f}, test_acc : {test_acc:.4f}')