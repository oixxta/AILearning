"""
쓰레기 분류기 만들기
"""
import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import mobilenet_v2

SEED = 42
random.seed = SEED
np.random.seed = SEED
tf.random.set_seed = SEED

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#print(BASE_DIR)
DATA_DIR_TRAIN = os.path.join(BASE_DIR, 'data', 'train')
#print(DATA_DIR_TRAIN)
DATA_DIR_VAL = os.path.join(BASE_DIR, 'data', 'val')
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 30
LR = 1e-3   #Learning Rate

### 학습용 데이터 세트
trainDs = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=SEED
)
### 검증용 데이터 세트
valiDs = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False,
    seed=SEED
)

classNames = trainDs.class_names
numClasses = len(classNames)
print('Classes : ', classNames, ' ', numClasses)    #Classes :  ['etc', 'glass', 'metal', 'paper', 'plastic']   5
#폴더명을 그대로 레이블로 사용함.

AUTOTUNE = tf.data.AUTOTUNE    #텐서플로우 최적의 병렬화 유지

trainDs = trainDs.cache().shuffle(1000).prefetch(AUTOTUNE)   #캐시메모리 사용, 디렉토리에 있던 이미지 파일들이 30개씩 캐시메모리로 입장. CPU와 GPU의 효율성을 더 높혀줌.
valiDs = valiDs.cache().prefetch(AUTOTUNE)

#데이터 변형 또는 증강
dataAug = keras.Sequential([
    layers.RandomFlip('horizontal'),    #좌우반전
    layers.RandomRotation(0.05),        #소량 회전
    layers.RandomZoom(0.1),             #소량 줌
])

preprocess = mobilenet_v2.preprocess_input    #[-1, 1] 범위 스케일링

#사전학습된 모델(mobileNetV2, 백본) 불러오기
base = mobilenet_v2.MobileNetV2(
    include_top=False,              #기본 분류기 미사용
    weights='imagenet',
    input_shape=IMG_SIZE + (3,),
)
base.trainable = False

#모델 설계 및 생성 (Functional API)
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = dataAug(inputs)         #입력에 증강 적용
x = layers.Lambda(preprocess)(x)
x = base(x, training = False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(units=numClasses, activation='softmax')(x)
model = keras.Model(inputs, outputs)

#모델 학습하기
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_accuracy', mode='max',
        save_best_only=True, verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max',
        patience=3, restore_best_weights=True
    ),
]
history = model.fit(
    trainDs, validation_data=valiDs, epochs=EPOCHS, callbacks=callbacks, verbose=2
)
val_loss, val_acc = model.evaluate(valiDs, verbose=0)
print(f'acc : {val_acc:.4f}, loss = {val_loss:.4f}')


#모델 파인튜닝 사용
unfreeze_from = 100

for layer in base.layers[unfreeze_from:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())
fine_history = model.fit(
    trainDs, validation_data=valiDs, epochs=EPOCHS, callbacks=callbacks, verbose=2
)
val_loss, val_acc = model.evaluate(valiDs, verbose=0)
print(f'fine_acc : {val_acc:.4f}, fine_loss = {val_loss:.4f}')

with open('class_name.txt', 'w', encoding='utf-8') as f:
    for name in classNames:
        f.write(f'{name}\n')


