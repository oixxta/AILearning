"""
문제6) 이미지 분류

CIFAR-100 dataset 사용    ( 얘 말고 다른 데이터를 이용해도 됨. 음식, 사무용 집기, 라면(우동) ... )
특징
 - 클래스 수: 100개 (예: 사과, 버스, 산, 고래, 시계 등)
 - 샘플 수: 60,000장, 학습용(train): 50,000장, 테스트용(test): 10,000장
 - 이미지 크기: 32×32 RGB (작은 해상도)
 - 레이블 구조: 100개 fine labels (세부 클래스), 20개 coarse labels (상위 클래스 그룹)

기본 CNN으로도 학습 가능하지만, 성능을 높이려면
 - 데이터 증강(ImageDataGenerator / tf.image)
 - 전이학습(사전학습 모델)
 - 정규화/드롭아웃/배치정규화 등을 함께 쓰는 게 효과적


-- 전체 흐름 요약 --
  작업1 :  CIFAR-100 dataset  분류 모델 작성 (MovileNetV2 모델로 전이학습, 파인튜닝)
  작업2 : 작성한  분류 모델 사용

              웹 브라우저에서 이미지 선택 
                   → 장고 웹서버에 저장 → 서버 내부에서 시각화로 확인(matplotlib) + 딥러닝 분류  
                  → 클라이언트에 분류 결과만 반환하기

작업2를 좀더 구체적으로 보면
 1) 클라이언트 
    : index.html에서 파일선택 버튼을 눌러 로컬 컴퓨터의 이미지 파일을 선택하고 화면에 선택된 이미지 출력
    : 분류결과요청 버튼 클릭 → AJAX 전송 (axios 모듈 사용)
 2) 서버(Django)
    : 수신된 이미지 파일 저장 → PIL + Matplotlib(imshow)으로 확인 → 딥러닝 분류 모델로 추론
    : 응답(JSON): 분류 결과만 반환(예 : bus)
 3) 클라이언트
    : 기존 이미지 아래에 이미지 분류 결과 문자열을 화면에 출력
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit


### cifar100 데이터 가져오기 및 확인
(xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='fine')
print('xTrain 샘플 수 : ', xTrain.shape)           #샘플 수 :  (50000, 32, 32, 3)
print('yTrain 샘플 수 : ', yTrain.shape)           #샘플 수 :  (50000, 1)
print('채널 수 : ', xTrain.shape[3])        #채널 수 :  3 (컬러)
print('이미지 크기 : ', xTrain.shape[1], xTrain.shape[2])   #이미지 크기 :  32 32
print('test 샘플 수 : ', xTest.shape)       #test 샘플 수 :  (10000, 32, 32, 3)
print('test 타입 수 : ', xTest.dtype)       #test 타입 수 :  uint8
print('레이블 종류의 갯수 : ', np.unique(yTrain).size)  #레이블 종류의 갯수 :  100


### Validation Data 만들기
yTrain_flat = yTrain.ravel()
sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)
(train_idx, vali_idx), = sss.split(xTrain, yTrain_flat)
xTrain_tr, yTrain_tr = xTrain[train_idx], yTrain[train_idx]   # (40000, 32, 32, 3), (40000, 1)
xVali, yVali = xTrain[vali_idx],  yTrain[vali_idx]
xTrain, yTrain = xTrain_tr, yTrain_tr
print('xVali 샘플 수 : ', xVali.shape)   # xVali 샘플 수 :  (10000, 32, 32, 3)
print('yVali 샘플 수 : ', yVali.shape)   # yVali 샘플 수 :  (10000, 1)


### 데이터 증강
imgGen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)
augSize = 20000
randIdx = np.random.randint(xTrain.shape[0], size=augSize)
xAug = xTrain[randIdx].copy()
yAug = yTrain[randIdx].copy()

gen = imgGen.flow(
    xAug,
    yAug,
    batch_size=augSize,
    shuffle=False,
    seed=1,
)
xAug, yAug = next(gen)
xTrain = np.concatenate([xTrain, xAug], axis=0)
yTrain = np.concatenate([yTrain, yAug], axis=0)
print('증강 후 xTrain 샘플 수 : ', xTrain.shape)   #증강 후 xTrain 샘플 수 :  (60000, 32, 32, 3)
print('증강 후 yTrain 샘플 수 : ', yTrain.shape)   #증강 후 yTrain 샘플 수 :  (60000, 1)


### 데이터 전처리
# x 이미지 리사이징(160 * 160으로), x 이미지 정규화, y 원핫 인코딩은 안함.(나중에 sparse 사용)
IMG_SIZE = (160, 160)
BATCH_SIZE = 64
NUM_CLASSES = 100       # np.unique(yTrain).size

def preprocess(x, y):
    x = tf.image.resize(x, IMG_SIZE)           # uint8 -> float32로 변환됨
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)     # [-1, 1] 스케일
    y = tf.squeeze(y, axis=-1)                 # (N,1) -> (N,)
    return x, y

trainDs = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)) \
    .shuffle(10000) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

testDs = tf.data.Dataset.from_tensor_slices((xTest, yTest)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

valiDs = tf.data.Dataset.from_tensor_slices((xVali, yVali)) \
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


### 사전학습된 모델(mobileNetV2, 백본) 불러오기
baseModel = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
baseModel.trainable = False


### 모델 설계 및 학습하기 : Sequential API 사용
model = tf.keras.Sequential([
    baseModel,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(trainDs, validation_data=valiDs, epochs=10, verbose=1)

loss, acc = model.evaluate(testDs, verbose=0)
print('최종 분류 손실도 : ', loss)
print('최종 분류 정확도 : ', acc)


### 모델 파인튜닝 사용
baseModel.trainable = True
print('베이스 모델의 레이어 수 : ', len(baseModel.layers))  #154
fineTuneAt = 100

for layer in baseModel.layers[:fineTuneAt]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint('mnv2_cifar100.keras', monitor='val_accuracy', save_best_only=True)
]

history2 = model.fit(trainDs, validation_data=valiDs, epochs=10, callbacks=cbs ,verbose=1)
loss, acc = model.evaluate(testDs, verbose=0)
print('최종 분류 손실도 : ', loss)
print('최종 분류 정확도 : ', acc)


### 최종 사용모델 모델파일로 저장
model.save('tfPractice6_CIFAR-100.keras') 