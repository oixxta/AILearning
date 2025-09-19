"""
데이터 증강(Data Augmentation)
CNN 모델의 성능을 높이고 오버 피팅을 극복하기 좋은 방법은 학습 이미지 데이터 양을 늘리는 것.
가장 쉽게 학습데이터의 양을 늘리는 방법은 데이터 증강으로, 개별 원본 이미지를 변형해서 학습하는 방법임.


"""
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

image = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

#augmentation이 적용된 image들을 시각화 해주는 함수
def show_aug_image(image, generator, n_images=4):
	
    # ImageDataGenerator는 여러개의 image를 입력으로 받기 때문에 4차원으로 입력 해야함.
    image_batch = np.expand_dims(image, axis=0)
	
    # featurewise_center or featurewise_std_normalization or zca_whitening 가 True일때만 fit 해주어야함
    generator.fit(image_batch) 
    # flow로 image batch를 generator에 넣어주어야함.
    data_gen_iter = generator.flow(image_batch)

    fig, axs = plt.subplots(nrows=1, ncols=n_images, figsize=(24, 8))

    for i in range(n_images):
    	#generator에 batch size 만큼 augmentation 적용(매번 적용이 다름)
        aug_image_batch = next(data_gen_iter)
        aug_image = np.squeeze(aug_image_batch)
        aug_image = aug_image.astype('int')
        axs[i].imshow(aug_image)
data_generator = ImageDataGenerator(horizontal_flip=True)
show_aug_image(image, data_generator, n_images=4)
