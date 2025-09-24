"""
전이학습(Transfer Learning) 실습

희귀한 소량의 이미지 데이터는 Cifar-10 데이터 세트 활용,
백본 모델로는 mobilenetv2 모델 구조 활용.

실습 1 : mobilenetv2 모델 그대로 학습시켜 내 이미지 데이터를 잘 분류하는 모델 생성 

mobilenetv2모델의 합성곱층과 분류층이 모두 학습에 참여하기 때문에 비효율적임 : 시간도 많이 걸리고 명중률도 저조함
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

### mobilenetv2 모델 호출
mobileNetModel = keras.applications.MobileNetV2(
    # mobilenetv2 : 입력 최소 크기 : 32, 권장 크기 : 96, 128, 160, 192, 224(기본값)
    input_shape = (32, 32, 3),
    include_top = True,         # 기본 분류기를 포함함
    weights=None,               # None or imagenet. imagenet으로 주면 이미지넷을 참조함.
    classes=num_classes,        # 내 이미지 데이터 클래스를 적용함. (10개)
)
#print(mobileNetModel.summary())
mobileNetModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = mobileNetModel.fit(xTrain, yTrain, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
print('test 평가 결과 : ', mobileNetModel.evaluate(xTest, yTest))