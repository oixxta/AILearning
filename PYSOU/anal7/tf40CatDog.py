"""
CNN을 활용해서 개와 고양이 이진분류

기존 라이브러리 활용

데이터에 이미지는 있지만, 레이블은 없음 > 레이블을 직접 만들 필요가 있음.(디렉토리명을 활용해서)
"""
import os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(1)   #시드넘버 고정
tf.random.set_seed(1)

### 데이터 가져오기
dataUrl = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zipPath = tf.keras.utils.get_file(
    fname='cats_and_dogs_filtered.zip',
    origin=dataUrl,
    extract=False,      #데이터 세트를 받자마자 압축풀기 금지
    cache_dir='.',
    cache_subdir='',
)
print(zipPath)

if not os.path.exists('./cats_and_dogs_filtered'):  #데이터 세트 압축 풀기
    with zipfile.ZipFile(zipPath, 'r') as obj:
        obj.extractall('.')
        print('Extract okay')

### 경로 정보를 확인
PATH = './cats_and_dogs_filtered'   #데이터 셋 루트(root)
trainDir = os.path.join(PATH, 'train')              #학습 폴더명의 경로
validationDir = os.path.join(PATH, 'validation')    #검증 폴더명의 경로
BATCH_SIZE = 128    #배치사이즈
EPOCHS = 20         #학습횟수
IMG_HEIGHT, IMG_WIDTH = 150, 150   #입력의 크기

trainCatsDir = os.path.join(trainDir, 'cats')     #학습폴더 안의 고양이 폴더
trainDogsDir = os.path.join(trainDir, 'dogs')     #학습폴더 안의 강아지 폴더
validationCatsDir = os.path.join(validationDir, 'cats')     #검증폴더 안의 고양이 폴더
validationDogsDir = os.path.join(validationDir, 'dogs')     #검증폴더 안의 강아지 폴더

### 데이터 확인
for p in [trainDir, trainCatsDir, trainDogsDir, validationDir, validationCatsDir, validationDogsDir]:
    print(p, '->', os.path.exists(p))   #폴더가 정상이면 True
print('cats(train) : ', len(os.listdir(trainCatsDir)), ', dogs(train) : ', len(os.listdir(trainDogsDir)))     #연습데이터 고앵이, 강아지 갯수 : 1000개씩
print('cats(val) : ', len(os.listdir(validationCatsDir)), ', dogs(val) : ', len(os.listdir(validationDogsDir))) #500개씩

### 제너레이터 준비(데이터 증강 및 스케일링)
trainIdg = ImageDataGenerator(   #학습데이터용 증강
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
valiIdg = ImageDataGenerator(   #검증데이터용 증강, 리스케일만 함.
    rescale=1./255
)    

trainData = trainIdg.flow_from_directory(   #flow_from_directory : 폴더(디렉토리)에 있는 것들에 사용 
    trainDir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',     #이진 분류이기에 binary 사용. categorical X
    shuffle=True,
)
valData = valiIdg.flow_from_directory( 
    validationDir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,
)

print('Class(label) index : ', trainData.class_indices) # 인덱스 확인 결과 : {'cats': 0, 'dogs': 1}
print('Class(label) index : ', valData.class_indices)   # 인덱스 확인 결과 : {'cats': 0, 'dogs': 1}

imgs, labels = next(trainData)
n_show = min(12, imgs.shape[0])
cols = 6
rows = int(np.ceil(n_show / cols))
idx_to_name = {v:k for k, v in trainData.class_indices.items()}
print(idx_to_name)      #{'cats': 0, 'dogs': 1}을 {0: 'cats', 1: 'dogs'}로 바꿈 (범주형 데이터 칼럼을 숫자화)

"""
### 데이터 세트 안의 이미지 중 임의의 개와 고양이 10개 시각화
plt.figure(figsize=(10, 2))
for i in range(n_show):
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(imgs[i])
    ax.set_title(f'{idx_to_name[int(labels[i])]}')
    ax.axis('off')
plt.suptitle('sample train images', fontsize=14)
plt.tight_layout()
plt.show()
plt.close()
"""

### 모델 설계(Sequential API 사용) 및 훈련
model = Sequential([
    Input((IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(units=128, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=1, activation='sigmoid'),
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

os.makedirs('chkpoints', exist_ok=True)
ck = ModelCheckpoint(
    filepath='chkpoints/catdogmodel.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=2
)
es = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

history = model.fit(trainData, epochs=EPOCHS, validation_data=valData, callbacks=[ck, es], verbose=2)
val_loss, val_acc = model.evaluate(valData, verbose=0)
print(f'acc : {val_acc:.4f}, loss : {val_loss:.4f}')


### 평가에 대한 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.show()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid()

plt.show()
plt.close()


### 모델로 예측해보기(검증 배치 예측)
previewGen = ImageDataGenerator(rescale=1./255)
previewFlow = previewGen.flow_from_directory(
    validationDir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=24,
    class_mode='binary',
    shuffle=True,
    seed=123
)
#예측용 개 / 고양이 6장 읽기
n_each = 6  # 고양이 / 개 각각 n개 모일 때까지 여러 배치 이어 받기.
catsImgs, dogsImgs = [], []
while len(catsImgs) < n_each or len(dogsImgs) < n_each:
    imgs, labels = next(previewFlow)    # 섞인 배치 자료
    for im, lb in zip(imgs, labels.ravel()):
        if lb == 0 and len(catsImgs) < n_each:
            catsImgs.append(im)
        elif lb == 1 and len(dogsImgs) < n_each:
            dogsImgs.append(im)
        if len(catsImgs) >= n_each and len(dogsImgs) >= n_each:
            break
#예측
catsProbs = model.predict(np.array(catsImgs), verbose=0).ravel()
dogsProbs = model.predict(np.array(dogsImgs), verbose=0).ravel()
print(catsProbs)
print(dogsProbs)


### 예측 결과 시각화
rows, cols = 2, n_each
plt.figure(figsize=(3 * cols, 5))
for i in range(n_each):
    #고앵이 줄
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(catsImgs[i])
    ax.axis('off')
    p = catsProbs[i]
    ax.set_title(f"True : cats | pred : {'dogs' if p >= 0.5 else 'cats'} (p_dog={p:.2f})", fontsize=9)
    #강아지 줄
    ax = plt.subplot(rows, cols, i + 1 + 1)
    ax.imshow(dogsImgs[i])
    ax.axis('off')
    p = dogsProbs[i]
    ax.set_title(f"True : dogs | pred : {'dogs' if p >= 0.5 else 'cats'} (p_dog={p:.2f})", fontsize=9)
plt.suptitle('validation preview', fontsize=12)
plt.tight_layout()
plt.show()
plt.close()


### 새 이미지를 분류 예측
import json

MODEL_PATH = 'chkpoints/catdogmodel.keras'      #기존에 만든 모델 읽어오기
THRESH = 0.5                                    #임계값

idx_to_name = {0 : 'cats', 1 : 'dogs'}
modelFromOutside = tf.keras.models.load_model(MODEL_PATH)
modelFromOutside.summary()

#전처리
def preprocessImg(img_path):
    #단일 이미지 경로를 받아 (1, 150, 150, 3) 텐서로 변환하기
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = tf.keras.utils.img_to_array(img)  # (H, W, Channel) float32
    arr = arr / 255.0                       # 스케일링
    arr = np.expand_dims(arr, axis=0)       # 차원 추가 -> (1, H, W, Channel)
    return arr

def predictOneImg(img_path, show=True):
    #이미지 하나를 분류예측하고 출력 후 반환.
    x = preprocessImg(img_path)
    probDog = float(modelFromOutside.predict(x, verbose=0)[0][0])   #시그모이드 출력 : 강아지일 확률
    predIdx = int(probDog >= THRESH)    #임계값 기준으로 이진화
    predName = idx_to_name[predIdx]
    probCat = 1.0 - probDog                                         #고양이일 확률. (1 - 강아지일 확률)

    #단일 이미지 시각화
    if (show == True):
        img_disp = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        plt.figure(figsize=(4, 4))
        plt.imshow(img_disp)
        plt.axis('off')
        plt.title(f'Pred: {predName} | p(cat)={probCat:.2f}, p(dog)={probDog:.2f}')
        plt.show()

    return {'path': img_path, 'pred': predName, 'p_dog':probDog}

result = predictOneImg('myImage.jpg', show=True)
print('result : ', result)