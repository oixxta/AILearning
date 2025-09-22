"""
fashion MNIST 데이터 세트 사용해서 이미지 보강 연습

텐서플로우가 지원하는 CNN 사용, 이미지 보강(ImageDataGenerator) 사용

이미지 보강을 지나치게 많이할 경우, 오히려 노이즈가 증가해서 정확도를 떨어트릴 수 있음. ex) 이미지가 지나치게 확대 등.
따라서 데이터가 충분히 많다면, 일부러 증강을 하는 것은 좋지 않음.
"""
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(0)           #랜덤 시드 고정
tf.random.set_seed(3)       #랜덤 시드 고정

# 데이터 가져오기 및 가공
(xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
xTrain = xTrain.reshape(-1, 28, 28, 1).astype('float32') / 255  #CNN이 선호하는 형태로 가공
xTest = xTest.reshape(-1, 28, 28, 1).astype('float32') / 255
#print(xTrain)
#print(yTrain)
yTrain = to_categorical(yTrain) #원핫 인코딩 적용
yTest = to_categorical(yTest)
#print(yTrain)
#print(yTest)


#이미지 시각화
plt.figure(figsize=(10, 10))
for c in range(100):
    plt.subplot(10, 10, c + 1)
    plt.axis('off')
    plt.imshow(xTrain[c].reshape(28, 28), cmap='gray')
plt.show()      #흑백 옷 데이터 100개 확인


#이미지 보강 작업
print(xTrain.shape) # 원본 자료 수 : (60000, 28, 28, 1)
print(yTrain.shape) # (60000, 10)
"""
imgGen = ImageDataGenerator(    #ImageDataGenerator 객체 생성
    rotation_range = 10,        #이미지 회전 : 랜덤하게 그림 0 ~ 180도 회전
    zoom_range = 0.1,           #이미지 확대 / 축소 : 10%
    shear_range = 0.5,          #축을 중심으로 전환 (모양 기울이기)
    width_shift_range = 0.1,    #수평이동
    height_shift_range = 0.1,   #수직이동
    horizontal_flip = True,     #좌우 수평 전환
    vertical_flip = False,      #상하 수직 전환
)       
augument_size = 100     # 원본 한개로 만들 증강 샘플 수 : 100개
idx = np.random.randint(xTrain.shape[0], size=augument_size)    #6만개 중 100개만 샘플로 뽑음
#print(idx)
xSrc = xTrain[idx].copy()    #원본 이미지 중 임의의 100개 복사한 것들을 저장
ySrc = yTrain[idx].copy()
#print(xSrc)
#print(ySrc)
gen = imgGen.flow(  #flow : RAM에 저장된 것을 가져옴. 보조기억장치(flow_from_directory)나 데이터 프레임(flow_from_dataframe)의 것들은 다른것을 선택.
    xSrc,
    y = np.zeros(augument_size),    # ySrc, 도 가능
    batch_size=augument_size,
    shuffle=False,                  # 순서를 섞지 않음
    seed=42,
)
xAugumented = next(gen)[0]          # flow 반환값은 제너레이터 객체임, 그래서 next() 다음 배치를 꺼내옴.
# 필요하면 원본에 합치기
xTrainAug = np.concatenate([xTrain, xAugumented], axis=0)
yTrainAug = np.concatenate([yTrain, ySrc], axis=0)
print(xTrainAug.shape)


#확인용 시각화
n = 16  #16개만
fig, axes = plt.subplots(1, n, figsize=(n, 4))
for i, ax in enumerate(axes):
    ax.imshow(xAugumented[i].squeeze(), cmap='gray')   #squeeze : 차원 감소 필요, 1차원으로
    ax.axis('off')
plt.tight_layout()
plt.show()
"""

imgGen = ImageDataGenerator(
    rotation_range = 10,        
    zoom_range = 0.1,
    shear_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False,
)
augumentSize = 30000    # 원본 한개로 만들 증강 샘플 수 : 30000개
randIdx = np.random.randint(xTrain.shape[0], size=augumentSize)
xAugument = xTrain[randIdx].copy()
yAugument = yTrain[randIdx].copy()

gen = imgGen.flow(
    xAugument,
    yAugument,
    batch_size=augumentSize,
    shuffle=False,
    seed=42,
)
xAugument, yAugument = next(gen)
#원본에 합치기
xTrain = np.concatenate([xTrain, xAugument], axis=0)
yTrain = np.concatenate([yTrain, yAugument], axis=0)
print(xTrain.shape)     # (90000, 28, 28, 1), 3만개 늘어남
print(yTrain.shape)     # (90000, 10)


# CNN 모델 설계하기(Sequential API로)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),               #입력 레이어

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),  #데이터 특징추출 레이어 1
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.1),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),  #데이터 특징추출 레이어 2
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.1),

    tf.keras.layers.Flatten(),                                  #1차원화 레이어

    tf.keras.layers.Dense(units=128, activation='relu'),        #분류기 레이어 1
    tf.keras.layers.Dropout(rate=0.3),

    tf.keras.layers.Dense(units=64, activation='relu'),         #분류기 레이어 2
    tf.keras.layers.Dropout(rate=0.3),

    tf.keras.layers.Dense(units=10, activation='softmax'),      #출력 레이어
])

# 모델 만들기
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


# 모델 최적화 설정
MODEL_DIR = './mnist/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath = './mnist/{epoch:02d}-{val_loss:.2f}.keras'
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor='val_loss', patience=5)


# 모델 학습시키기
history = model.fit(xTrain, yTrain, epochs=100, batch_size=64, validation_split=0.2, callbacks=[chkpoint, earlystop], verbose=2)


# 모델 정확도 확인
print('Test accuracy : %.4f'%(model.evaluate(xTest, yTest)[1]))     # Test accuracy : 0.9234


# 모델 시각화 확인
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], marker='o', c='red', label='acc')
plt.plot(history.history['val_accuracy'], marker='s', c='blue', label='val_acc')
plt.xlabel('epochs')
plt.ylim(0.3, 1)
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], marker='o', c='red', label='loss')
plt.plot(history.history['val_loss'], marker='s', c='blue', label='val_loss')
plt.xlabel('epochs')
plt.ylim(0.0, 1)
plt.legend(loc='upper right')

plt.show()