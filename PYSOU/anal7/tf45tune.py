"""
전이학습을 활용한 다항 분류 실습

꽃(클래스 : 5개) 분류

희귀한 소량의 이미지 데이터는 텐서플로우 데이터 세트에 있는 꽃 데이터 사용,
백본 모델로는 mobilenetv2 모델 구조 활용.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

### 데이터 가져오기
(trainDs, validationDs), dsInfo = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'], #8:2로 데이터를 나눔.
    with_info=True,
    as_supervised=True,          # True : 튜플타입, False : 딕트 타입
    shuffle_files=True           # 파일 섞기 : True
)

### 데이터 체크하기
print(trainDs)
print(validationDs)
#print(metadata)     #데이터 세트의 정보를 출력함.
total = dsInfo.splits['train'].num_examples
print('train 원본 전체 데이터 갯수 : ', total)              #train 원본 전체 데이터 갯수 :  3670
print('trainDs 데이터 갯수 : ', int(total * 0.8))          #trainDs 데이터 갯수 :  2936
print('validationDs 데이터 갯수 : ', int(total * 0.2))     #validationDs 데이터 갯수 :  734
#샘플 크기 확인
for image, label in trainDs.take(10):
    print('원본 1장 : ', image.shape, label.numpy())     #원본 1장 : (333, 500, 3) 2, 사진 크기들이 제각각임.
#레이블 확인
get_label_name=dsInfo.features['label'].int2str
print(get_label_name(4))    # roses
print(get_label_name(3))    # sunflowers
print(get_label_name(2))    # tulips
print(get_label_name(1))    # daisy
print(get_label_name(0))    # dandelion
print(dsInfo.features['label'].names)
"""
#이미지 한 장 시각화로 확인해보기
for image, label in trainDs.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.axis('off')
    plt.show()
"""


### 데이터 전처리
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0      #0~1 사이로 정규화
    return image, label
#AUTOTUNE : GPU 코어 갯수 / 리소스 상황에 맞게 자동으로 최적화
trainDs = trainDs.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validationDs = validationDs.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


### 사전학습된 모델(mobilenetv2, 백본) 불러오기
baseModel = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),    #(160, 160, 3)
    include_top=False,              # 기본 분류기를 포함하지 않음, 컨볼루션(합성곱층)만 남음.
    weights='imagenet'              # 전이학습을 쓸것이기 때문에 이미지 넷을 넣어야 성능이 개선됨.
)
baseModel.trainable = False         #모델 동결


### 모델 설계하기 : Sequantial API 사용
model = tf.keras.Sequential([
    baseModel,                      #위에서 정의한 수정된 백본 가져오기
    tf.keras.layers.GlobalAveragePooling2D(),   #특징맵 (featureMap)을 평탄화 함.
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=dsInfo.features['label'].num_classes, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(
    trainDs, validation_data=validationDs, epochs=5, verbose=1
)
loss, acc = model.evaluate(validationDs, verbose=0)
print('최종 분류 손실도 : ', loss)
print('최종 분류 정확도 : ', acc)



### Fine-Tunning(미세조정) 실시, 전이학습 후 성능 향상을 목적으로.
baseModel.trainable = True
print('total layers : ', len(baseModel.layers))     #baseModel 전체 레이어 갯수 : 154개
for i, layer in enumerate(baseModel.layers):        #baseModel 전체 레이어 목록 출력
    if layer.trainable:
        print(f'[{i:03}] {layer.name}')

fine_tune_at = 100

for layer in baseModel.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
model.fit(trainDs, validation_data=validationDs, epochs=5, verbose=1)
loss, acc = model.evaluate(validationDs, verbose=0)
print('튜닝 후 최종 분류 손실도 : ', loss)
print('튜닝 후 최종 분류 정확도 : ', acc)


### 미세조정된 모델로 예측 한번 해보기
for image, label in validationDs.take(1):
    sample_images = image
    sample_labels = label
    break

pred_probs = model.predict(sample_images)
#print('pred_probs : ', pred_probs)
pred_classes = tf.argmax(pred_probs, axis=1)
print('pred_classes : ', pred_classes)
class_names = dsInfo.features['label'].names
print('class_names : ', class_names)
#예측 인덱스 vs 실제 인덱스 출력 코드
for i in range(len(sample_images)):
    predicted_index = int(pred_classes[i])
    actual_index = int(sample_labels[i])
    predicted_name = class_names[predicted_index]
    actual_name = class_names[actual_index]
    print(f'[{i:02}] Predicted : {predicted_index}({predicted_name}) | Actual : {actual_index}({actual_name})')
#시각화
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(sample_images[i])
    predicted_label = class_names[pred_classes[i]]
    actual_label = class_names[sample_labels[i]]
    color = 'blue' if predicted_label == actual_label else 'red'
    plt.title(f'Predicted : {predicted_label}\nActual : {actual_label}', color=color, fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.close()