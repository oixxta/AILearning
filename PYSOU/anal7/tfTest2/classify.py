"""
학습이 끝난 최고의 모델을 가져와서 분류해보기

"""
import os, json, random, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import mobilenet_v2

IMG_SIZE = (224, 224)

def load_class_names(path='class_name.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

#모델이 기대하능 입격차원과 혀익을 맞춰주를 전처리함]]
def load_and_preprocess(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)  #ㅇ이미지를 float  ㅕㄹ대톨 출력
    np.expand_dims(arr, axis=0)              #배열 차원 추가
    return arr


def main():
    if len(sys.argv) < 2:
        print('분류할 파일명.확장자 입력 : ')
        sys.exit(1)                          #비정상 종료 코드로 종료
    
    image_path = sys.argv[1]
    print(image_path)

    # 이미지 분류 모델 로딩하기
    model = keras.models.load_model(
        'best_model.keras',
        compile=False,
        custom_objects={'preprocess_input' : mobilenet_v2.preprocess_input}
    )

    # 'class_name.txt'를 읽어서 인덱스를 클래스명과 맵핑
    class_names = load_class_names(r'C:\Users\msi\Desktop\Jongseong KIM\AILearning\PYSOU\anal7\tfTest2\class_name.txt')
    print(class_names)

    # 임력 이미지 전처리
    x = load_and_preprocess(image_path)
    preds = model.predict(x, verbose=0)[0]
    print(preds)
    topIdx = int(np.argmax(preds))
    topProb = float(preds[topIdx])

    # 1 ~ 3위까지 클래스 출력하기
    print(f'예측값 : {class_names[topIdx]} (확률 : {topProb:.3f})')
    order = np.argsort(-preds)  #내림차순 정렬 인덱스
    print('분류 예측 결과 : ')
    for i in order[:3]:
        print(f'{class_names[i]:} {preds[i]:.3f}')


if __name__ == '__main__':
    main()

# 실행은 > python classify.py 파일명.jpg 로 저장.