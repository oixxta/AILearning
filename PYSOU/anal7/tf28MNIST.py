"""
TF27에서 만든 모델을 불러와서 TF26에서 불러와본 내가 만든 숫자 이미지를 분류시켜서 예측 시켜보기
"""

import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


myModel = keras.models.load_model('tf27model.keras')

im = Image.open('su.png')
#원래 이미지 크기를 28 * 28 픽셀로 해야 함. MNIST 데이터 세트 기준에 맞추기 위해. 리사이즈.
#컬러 이미지의 경우, 흑백(0~255)으로 변환 후 numpy 배열로 전환 필요함.
img = np.array(im.resize((28, 28), Image.Resampling.LANCZOS).convert('L'))

print(img.shape)

plt.imshow(img, cmap='Greys')
plt.show()
plt.close()

# (28 * 28)이미지를 (1, 784) 벡터로 변환하기 (Dense 클래스 입력 형태)
data = img.reshape([1, 784]).astype('float32')
print(data)

# 정규화 실시 (Dense는 정규화 된 실수를 선호함.)
data = data / 255.0     # 픽셀 값들을 0 ~ 1 범위로 정규화함.
print(data)

# 모델로 숫자 예측 시켜보기
pred = myModel.predict(data)
print('pred : ', pred)
print('예측값 : ', np.argmax(pred, 1))

# 다시 시각화 (1, 784)를 (28 * 28)로 변환 (reshape)
plt.imshow(data.reshape(28 , 28), cmap = 'Greys')
plt.show()
plt.close()
