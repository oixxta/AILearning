"""
MNIST 데이터 세트로 CNN 모델을 작성해보기

Sequantial API 사용

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


#모델 정의하기 : Sequantial API
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),    #MaxPool과 MaxPooling은 하는 일이 다름.
    tf.keras.layers.Dropout(rate=0.2),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),   #FCLayer (Fully-Connected Layer, 차원을 1차원으로 감소시킴.)
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(rate= 0.3),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(rate= 0.2),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])
print(model1.summary())

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model1.fit(x_Train, y_Train, epochs=100, batch_size=128, validation_split=0.1, callbacks=[es], verbose=2)

#모델 평가
train_loss, train_acc = model1.evaluate(x_Train, y_Train, verbose=0)
test_loss, test_acc = model1.evaluate(x_Test, y_Test, verbose=0)
print(f'train_loss : {train_loss:.4f}, train_acc : {train_acc:.4f}')
print(f'test_loss : {test_loss:.4f}, test_acc : {test_acc:.4f}')

#모델 저장
save_path = "MNIST_cnn.keras"
model1.save(save_path)

#모델 읽기
loaded_model = tf.keras.models.load_model(save_path)
loss2, acc2 = loaded_model.evaluate(x_Test, y_Test, verbose=0)
print(f'loss2 : {loss2:.4f}, acc2: {acc2:.4f}')

#기존 자료도 1개로 예측
idx = 0
x_one = x_Test[idx:idx + 1]
y_true = int(y_Test[idx])
probs = loaded_model.predict(x_one, verbose=0)[0]
y_pred = int(np.argmax(probs))
print(f'실제값 : {y_true}, 예측값 : {y_pred}, 확률값 : {np.round(probs, 3)}')

#시각화 하기 : 학습 곡선 보기(정확도, 손실)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

#단일 이미지 + 예측 확률 막대그래프 시각화 하기
classes = [str(i) for i in range(10)]
print(classes)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_one[0].squeeze(), cmap='gray')
plt.axis('off')
plt.title(f'True : {y_true}, | Pred : {y_pred}')

plt.subplot(1, 2, 2)
plt.bar(classes, probs)
plt.title('Prediction Probabilites')
plt.xlabel('class')
plt.ylabel('Probability')
plt.ylim(0, 1.0)

for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()
plt.close()

#컨퓨전 매트릭스 확인
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred_all = np.argmax(loaded_model.predict(x_Test, verbose=0), axis=1)
print(y_pred_all)
cm = confusion_matrix(y_Test, y_pred_all, labels=list(range(10)))
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title('confusion matrix')
plt.tight_layout()
plt.show()