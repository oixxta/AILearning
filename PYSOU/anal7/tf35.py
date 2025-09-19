"""
이항분류 : 남자 사진들과 여자 사진들(persin_img 폴더)을 학습해 모델을 만듬.

CNN : 성인 남녀 얼굴이미지 분류
"""
import cv2, os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

img_dir = 'person_img'
xData, yData = [], []

#남녀 구분 라벨 구하기 - 파일명에서 추출 - split 함수

for file in os.listdir(img_dir):
    try:
        gender = file.split('_')    #0 : male, 1 : female
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64)) # 크기 축소
        xData.append(img)
        yData.append(gender)
        #print(xData)
        #print(yData)
    except:
        continue