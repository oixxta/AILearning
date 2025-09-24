"""
전이학습(Transfer Learning) 실습

희귀한 소량의 이미지 데이터는 Cifar-10 데이터 세트 활용,
백본 모델로는 mobilenetv2 모델 구조 활용.

실습 2 : mobilenetv2 모델 그대로 학습시켜 내 이미지 데이터를 잘 분류하는 모델 생성, 
추가로 전이학습(Transfer Learning)과 미세조정(Fine-Tunning)도 실시
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers

### 데이터 가져오기 및 전처리
(xTrain, yTrain), (xTest, yTest) = keras.datasets.cifar10.load_data()
print(xTrain.shape)     #(50000, 32, 32, 3)
num_classes = 10        #내 희귀한 데이터는 레이블이 10개.

xTrain = xTrain.astype('float32') / 255.0   #정규화
xTest = xTest.astype('float32') / 255.0     #정규화
yTrain = keras.utils.to_categorical(yTrain, num_classes)    #원 핫 인코딩
yTest = keras.utils.to_categorical(yTest, num_classes)      #원 핫 인코딩
print('train data : ', xTrain.shape, yTrain.shape)      #train data :  (50000, 32, 32, 3) (50000, 10)
print('test data : ', xTest.shape, yTest.shape)         #test data :  (10000, 32, 32, 3) (10000, 10)


### 전이학습(Transfer Learning) : 기존 모델(백본) 중 일부만 불러와서 학습에 참여시킴.
#기존 모델의 가중치는 모두 동결(freeze) -> 새로 추가한 분류층만 학습시킴.
base_model = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,      # 기본 분류기를 포함하지 않음, 컨볼루션(합성곱층)만 남음.
    weights='imagenet',     # 전이학습을 쓸것이기 때문에 이미지 넷을 넣어야 성능이 개선됨.
)
base_model.trainable = False    # 모델의 기존 가중치를 모두 동결시킴 : 
#결과적으로 MobileNetV2의 컨볼루션만 사용함. 분류기는 직접 정의할 것임.

### Functional API로 모델 만들기 : MobileNetV2 컨볼루션을 재활용
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Resizing(96, 96)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)    #MaxPooling보다 더 급격하게 feature의 크기를 줄임.
outputs = layers.Dense(units=num_classes, activation='softmax')(x)     #내가 직접 정의한 분류기

model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=5, batch_size=64, validation_split=0.3, verbose=1)
print('test 평가 결과 : ', model.evaluate(xTest, yTest))    #test 평가 결과 :  [0.5890288352966309, 0.7953000068664551]


### Fine-Tunning (미세조정)
# 동결되어 있던 합성곱 층 중 일부만 동결에서 해제해서 학습에 참여시킴. 대체적으로 성능을 개선함.
# 베이스 모델 일부 층만 열기 (예 : 마지막 30개만)
base_model.trainable = True     #백본의 특징추출기 동결 해제
print(len(base_model.layers))           #전체 베이스 레이어 층의 갯수 확인
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 낮은 학습률로 재 컴파일 (optimizer를 건드려야 함. 기본값인 0.0001보다 작게)
# weight를 너무 변화를 주면 안됨.
model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(xTrain, yTrain, epochs=5, batch_size=64, validation_split=0.3, verbose=1)
print('test 평가 결과 : ', model.evaluate(xTest, yTest))    #test 평가 결과 :  [0.5145906209945679, 0.8300999999046326]