"""
문제5) 이미지 분류

Roboflow Public Datasets 사용
 - Rock Paper Scissors Classification Dataset : “바위/보/가위” 손 모양 이미지( 약 2,925장 ). 클래스도 3개, 컬러 이미지. 
   https://public.roboflow.com/classification/rock-paper-scissors?utm_source=chatgpt.com

** Roboflow에서 Rock Paper Scissors Classification Dataset 데이터 받기 (가장 쉬운 방법: ZIP 다운로드) **
  -  Roboflow Public 페이지에서 Rock Paper Scissors Classification Dataset의 Train/Valid/Test 압축파일을 내려받는다. 
  - 아래처럼 풀어줌(예시 경로):

data/rock-paper-scissors/
 ├─ train/
 │   ├─ rock/       *.jpg, *.png ...
 │   ├─ paper/
 │   └─ scissors/
 ├─ valid/
 │   ├─ rock/
 │   ├─ paper/
 │   └─ scissors/
 └─ test/
     ├─ rock/
     ├─ paper/
     └─ scissors/
위와 같이 클래스별 하위 폴더 구조만 맞으면 Keras가 자동으로 라벨을 매겨 준다.
  - Keras로 불러오기 + feature/label 일부 출력
  

마지막에 이미지 증강 전,후의 모델 성능 비교 - ROC curve 사용
새로운 이미지에 대한 분류 결과 확인
"""

import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)   # 필요에 따라 (128,128) 등으로 변경
BATCH    = 32

train_dir = "Rock Paper Scissors.v1-raw-300x300.folder/train"
valid_dir = "Rock Paper Scissors.v1-raw-300x300.folder/valid"
test_dir  = "Rock Paper Scissors.v1-raw-300x300.folder/test"

# 디렉터리에서 이미지 분류용 데이터셋 만들기
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",             # [0..C-1] 정수 라벨
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=42,
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=True,
    seed=42,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    shuffle=False
)
# 클래스 이름 확인
class_names = train_ds.class_names
print("class_names:", class_names)    # 예: ['paper', 'rock', 'scissors']
# --- feature와 label 일부 출력 ---
for images, labels in train_ds.take(1):
    print("features shape:", images.shape)   # (B, H, W, 3)
    print("labels shape:", labels.shape)     # (B,)
    print("labels (first 10):", labels[:10].numpy())
    print("labels mapped (first 10):", [class_names[i] for i in labels[:10].numpy()])

