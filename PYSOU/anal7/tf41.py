"""
패션 MNIST 데이터 세트로 CNN 처리 : 3가지 분류모델을 사용해 보기

실습 1 : Conv + Dense
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

### 데이터 가져오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
print(xTrain.shape, yTrain.shape)

### 데이터 정규화
xTrain = xTrain / 255.0
xTest = xTest / 255.0
#print(xTrain[0])
xTrain = xTrain.reshape(-1, 28, 28, 1)
xTest = xTest.reshape(-1, 28, 28, 1)
print(xTrain.shape, xTest.shape)

### 시각화
plt.figure(figsize=(10, 10))

for c in range(16):
    plt.subplot(4, 4, c + 1)
    plt.imshow(xTrain[c].reshape(28, 28), cmap='gray')
plt.show()
print(yTrain[:16])


### 실습 1 : Conv + Dense
def practiceOne():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model1.summary())
    history1 = model1.fit(xTrain, yTrain, epochs=15, validation_split=0.25, verbose=2)
    print(model1.evaluate(xTest, yTest, verbose=0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], 'b-', label='loss')
    plt.plot(history1.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['accuracy'], 'b-', label='accuracy')
    plt.plot(history1.history['val_accuracy'], 'r--', label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()


### 실습 2 : (Conv + Pooling) + Dense
def practiceTwo():
    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model1.summary())
    history1 = model1.fit(xTrain, yTrain, epochs=15, validation_split=0.25, verbose=2)
    print(model1.evaluate(xTest, yTest, verbose=0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], 'b-', label='loss')
    plt.plot(history1.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['accuracy'], 'b-', label='accuracy')
    plt.plot(history1.history['val_accuracy'], 'r--', label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()      #1번과 비교해 과적합이 크게 완화됨. 명중률은 1% 증가


### 실습 3 : 효율 향상을 위해 성능 좋은 기존 네트워크 일부를 도용 + (Conv + Pooling) + Dense
### AlexNet, VGGNet, GoogLeNet, ResNet, MobileNet 등..
### https://cafe.daum.net/flowlife/S2Ul/31
def practiceThree():
    # VGGNet 스타일 돚거
    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model1.summary())
    history1 = model1.fit(xTrain, yTrain, epochs=15, validation_split=0.25, verbose=2)
    print(model1.evaluate(xTest, yTest, verbose=0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], 'b-', label='loss')
    plt.plot(history1.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['accuracy'], 'b-', label='accuracy')
    plt.plot(history1.history['val_accuracy'], 'r--', label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()     #2번과 비교해 명중률 대폭 상승(20%, 최종 명중률 91%)


### 실습 4 : 효율 향상을 위해 성능 좋은 기존 네트워크 일부를 도용 + (Conv + Pooling) + Dense + 이미지 증강까지!
def practiceFour():
    # 이미지 데이터 증강
    idg = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        shear_range=0.5,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
    )
    augmentSize = 20000
    randidx = np.random.randint(xTrain.shape[0], size=augmentSize)
    xAugmented = xTrain[randidx].copy()
    yAugmented = yTrain[randidx].copy()
    xAugmented = idg.flow(xAugmented, np.zeros(augmentSize), batch_size=augmentSize, shuffle=False).next()[0]
    trainX = np.concatenate((xTrain, xAugmented))
    trainY = np.concatenate((yTrain, yAugmented))

    print(trainX.shape)     #2만개씩 늘어남
    print(trainY.shape)

    # VGGNet 스타일 돚거
    model1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model1.summary())
    history1 = model1.fit(trainX, trainY, epochs=15, validation_split=0.25, verbose=2)
    print(model1.evaluate(xTest, yTest, verbose=0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], 'b-', label='loss')
    plt.plot(history1.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['accuracy'], 'b-', label='accuracy')
    plt.plot(history1.history['val_accuracy'], 'r--', label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()     #3번과 비교해 명중률 대동소이, 시각화 그래프 결과로는 많이 단조로워짐. 기술적으로도 이미 데이터가 많으면 큰 차이는 없음.

practiceFour()