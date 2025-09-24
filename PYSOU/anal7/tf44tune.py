"""
MobileNetv2를 활용한 전이학습을 적용한 개/고양이 이진분류 모델 생성
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds  #텐서플로우 공개 데이터 세트
import tensorflow as tf

tfds.disable_progress_bar()

### 데이터 가져오기
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], #8:1:1로 데이터를 나눔.
    with_info=True,
    as_supervised=True          # True : 튜플타입, False : 딕트 타입
)

### 데이터 체크하기
print(raw_train)
print(raw_validation)
print(raw_test)
#print(metadata)     #데이터 세트의 정보를 출력함.
total = metadata.splits['train'].num_examples
print('train 원본 전체 데이터 갯수 : ', total)
print('raw_train 데이터 갯수 : ', int(total * 0.8))
print('raw_validation 데이터 갯수 : ', int(total * 0.1))
print('raw_test 데이터 갯수 : ', int(total * 0.1))
#샘플 크기 확인
for image, label in raw_train.take(10):
    print('원본 1장 : ', image.shape, label.numpy())     #원본 1장 :  (262, 350, 3) 1, 사진 크기들이 제각각임.
#레이블 확인
get_label_name=metadata.features['label'].int2str
print(get_label_name(1))    #dog
print(get_label_name(0))    #cat
"""
#이미지 한 장 시각화로 확인해보기
for image, label in raw_train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis('off')
    plt.show()
"""

### 데이터 전처리
imgSize = 160
def formatEx(image, label):
    image = tf.cast(image, tf.float32)  # unsigned int -> float 32로 형변환
    image = (image / 127.5) - 1.0
    image = tf.image.resize(image, (imgSize, imgSize))  #160 * 160으로 리사이즈
    return image, label

#AUTOTUNE : GPU 코어 갯수 / 리소스 상황에 맞게 자동으로 최적화
# GPU idle time을 최소화시킴.
train = raw_train.map(formatEx, num_parallel_calls=tf.data.AUTOTUNE)
validation = raw_validation.map(formatEx, num_parallel_calls=tf.data.AUTOTUNE)
test = raw_test.map(formatEx, num_parallel_calls=tf.data.AUTOTUNE)

for img, label in train.take(1):
    print('전처리 결과 type : ', img.dtype)
    print('전처리 결과 shape : ', img.shape)
    print('min/max : ', float(tf.reduce_min(img.shape)), float(tf.reduce_max(img.shape)))

### 배치 파이프라인 작성 (학습용 / 검증용)
# 1000개의 샘플을 메모리에 가져와 무작위로 섞음 -> 그 다음 데이터 버퍼에 읽어 또 섞음.
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
#train만 섞어줌
train_batches = (train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
validation_batches = (validation.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
test_batches = (test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

for image_single, label_single in raw_train.take(2):
    print('원본 단일 이미지 shape : ', image_single.numpy().shape)
    print('레이블 : ', label_single.numpy())


### 모델 설계하기
#베이스 모델 작업
IMG_SHAPE = (imgSize, imgSize, 3)
baseModel = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
baseModel.trainable = False

#전처리/batch된 텐서를 통과시켜 특징 맵 얻기.
images_batch, labels_batch = next(iter(train_batches))
feature_batch = baseModel(images_batch)
print('입력 배치 shape : ', images_batch.shape)     # 입력 배치 shape :  (32, 160, 160, 3)
print('특징맵 배치 shape : ', feature_batch.shape)  # 특징맵 배치 shape :  (32, 5, 5, 1280)

global_avg = tf.keras.layers.GlobalAveragePooling2D()(feature_batch)
print('GAP 이후의 shape : ', global_avg.shape)      # GAP 이후의 shape : (32, 1280)


#모델 정의하기 : Sequential API 사용
"""
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=IMG_SHAPE),
    baseModel,      #특징 추출기(컨볼루션) - 동결상태(갱신 안함)
    tf.keras.layers.GlobalAveragePooling2D(),   #GAP라고 줄여서 부름.
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])
"""

#모델 정의하기 : Functional API 사용
inputs = tf.keras.layers.Input(shape=IMG_SHAPE)
x = baseModel(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

baseModel.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_batches, validation_data=validation_batches, epochs=5, verbose=1
)

#모델 성능평가
test_loss, test_acc = model.evaluate(test_batches, verbose=0)
print('test_loss', test_loss, ' test_acc', test_acc)    #test_loss 0.049  test_acc 0.981

#history 확인
print(history.history.keys())

#학습곡선 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='train acc')
plt.plot(epochs_range, val_acc, label='validation acc')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='train loss')
plt.plot(epochs_range, val_loss, label='validation loss')
plt.legend(loc='upper right')

plt.show()
plt.close()

### Fine-Tunning (미세조정)
baseModel.trainable = True
print(len(baseModel.layers))           #전체 베이스 레이어 층의 갯수 확인
fine_tune_at = 100  #백본의 100번째 레이어까지만 동결 유지할 계획.
for layer in baseModel.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
chk_path_ft = 'finetune_best.keras'
callbalck_ft = [
    tf.keras.callbacks.ModelCheckpoint(
        chk_path_ft, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(   #val_loss 개선이 멈추면 lr을 0.5배 줄임.
        monitor='val_loss', factor=0.5, patience=2, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1
    )
]

EPOCHS_TRANSFER = 5
EPOCHS_FINETUNE = 5

# 전이학습이 끝난 후, 이어서 미세조정
history_ft = model.fit(
    train_batches, 
    validation_data=validation_batches, 
    epochs=EPOCHS_TRANSFER + EPOCHS_FINETUNE,
    initial_epoch=len(history.history['loss']),
    callbacks=callbalck_ft,
    verbose=2, 
)
test_loss, test_acc = model.evaluate(test_batches, verbose=0)
print('test_loss', test_loss, ' test_acc', test_acc) #test_loss 0.049  test_acc 0.986

# 전이학습 vs 미세조정 학습 곡선 결합한 시각화
def concat_hist_func(h1, h2):
    keys = h1.history.keys()
    out = {}
    for k in keys:
        out[k] = h1.history[k] + h2.history[k]
    return out

hist_all = concat_hist_func(history, history_ft)
acc = hist_all['accuracy']
val_acc = hist_all['val_accuracy']
loss = hist_all['loss']
val_loss = hist_all['val_loss']

epochs = range(1, len(acc) + 1)
split_epoch = EPOCHS_TRANSFER       # 전이학습과 미세조정 경계선 위치

plt.figure(figsize=(12, 5))
# 정확도 ---
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, marker='o', label='train acc')
plt.plot(epochs, val_acc, marker='s', label='val acc')
for i, v in enumerate(acc):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(val_acc):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=8)
plt.axvline(split_epoch, linestyle='--', alpha=0.6, label='Fine-tuning go')
plt.title('Accuracy (Transfer Learning -> Fine tune)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# 손실 ---
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, marker='o', label='train loss')
plt.plot(epochs, val_loss, marker='s', label='val loss')
for i, v in enumerate(loss):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(val_loss):
    plt.text(epochs[i], v, f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=8)
plt.axvline(split_epoch, linestyle='--', alpha=0.6, label='Fine-tuning go')
plt.title('Loss (Transfer Learning -> Fine tune)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
plt.close()