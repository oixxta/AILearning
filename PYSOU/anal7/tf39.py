"""
tf38에서 이어짐

CNN 사용
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = xTrain.astype('float32') / 255.0
xTest = xTest.astype('float32') / 255.0
yTrain = to_categorical(yTrain, 10)
yTest = to_categorical(yTest, 10)

#모델 만들기(functional)
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


inputs = Input(shape=(32, 32, 3))
#stage1
x = conv_block(inputs, 32)
x = conv_block(x, 32)
x = MaxPooling2D()(x)

#stage2
x = conv_block(x, 64)
x = conv_block(x, 64)
x = MaxPooling2D()(x)

#stage3
x = conv_block(x, 128)
x = conv_block(x, 128)
x = MaxPooling2D()(x)

# 분류기
from keras.layers import GlobalAveragePooling2D
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(units=10, activation='softmax')(x)

#모델 만들기
model = Model(inputs, outputs, name='CIFAR10_CNN')
print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True)
history = model.fit(xTrain, yTrain, batch_size=64, epochs=100, validation_split=0.1, shuffle=True, verbose=2, callbacks=es)

test_loss, test_acc = model.evaluate(xTest, yTest, verbose=0)
print('test_acc : ', test_acc)
print('test_loss : ', test_loss)


"""
이미지 처리에는 CNN 사용이 압도적으로 더 좋음. (비CNN : 50%, CNN : 97%)
"""