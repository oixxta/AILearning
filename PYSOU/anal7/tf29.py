"""
Fashion MNIST 데이터 셋으로 이미지 분류 모델 만들기
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape, ' ', train_labels.shape)  #(60000, 28, 28)   (60000,)
print(test_images.shape, ' ', test_labels.shape)    #(10000, 28, 28)   (10000,)
print(set(train_labels))    #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

plt.imshow(train_images[0], cmap='Greys')
plt.show()

# 25개의 이미지 확인
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
    plt.imshow(train_images[i])
plt.show()

# 데이터 전처리하기
train_images = train_images / 255.0     #정규화 및 실수화
test_images = test_images / 255.0
#print(train_images[0])

# 신결망 모델 구성하기
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),          #28 * 28을 748열 짜리로 바꿔줌.
    tf.keras.layers.Dense(units=64, activation='relu'),    #완전(밀집) 연결
    tf.keras.layers.Dense(units=32, activation='relu'),    #완전(밀집) 연결
    tf.keras.layers.Dense(units=10, activation='softmax')
])
print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense (Dense)               (None, 128)               100480

 dense_1 (Dense)             (None, 64)                8256

 dense_2 (Dense)             (None, 10)                650

=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
_________________________________________________________________
"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=128, epochs=5, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('정확도 : ', test_acc)    #정확도 :  0.8565000295639038
print('손실도 : ', test_loss)   #손실도 :  0.3916696310043335


# 만든 모델로 예측 해보기
pred = model.predict(test_images, verbose=0)
print(pred[0])
print('예측값 : ', np.argmax(pred[0]))
print('실제값 : ', test_labels[0])


# 각 이미지 출력용 함수 만들기 (예측 이미지와 실제 레이블을 비교해보기)
def plot_image(i, pred_arr, true_label, img):
    pred_arr, true_label, img = pred_arr[i], true_label[i], img[i]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='Greys')
    pred_label = np.argmax(pred_arr)
    # 예측값과 실제값이 같으면 blue, 다르면 red
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pred_label], 100 * np.max(pred_arr), class_names[true_label]), color=color)
    #맞으면 파랑, 틀리면 빨강으로

i = 20
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, pred, test_labels, test_images)

def plot_value_arr(i, pred_arr, true_label):
    pred_arr, true_label = pred_arr[i], true_label[i]
    thisplot = plt.bar(range(10), pred_arr)
    plt.ylim([0, 1])
    pred_label = np.argmax(pred_arr)
    thisplot[pred_label].set_color('red')   # 예측값
    thisplot[true_label].set_color('blue')  # 실제값

plt.subplot(1, 2, 2)
plot_value_arr(i, pred, test_labels)

plt.show()
plt.close()