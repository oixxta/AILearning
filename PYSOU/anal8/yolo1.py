"""
YOLO

객체 탐지(Object detection) 중 one-step 방식, 이미 학습이 끝난 모델(yolov8n.pt) 사용.
데이터는 cocos data를 사용함 : 대량의 데이터 처리에 용이
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#from ultralytics import YOLO
#
#try:
#    model = YOLO('yolov8n.pt')      #학습이 끝난 YOLO 모델 저장
#except Exception as e:
#    print('Error!')
#


# YOLO 임포팅
import subprocess
import sys

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print('ultralytics not installed! try to install...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics'])
    except subprocess.CalledProcessError as e:
        raise SystemExit('ultralytics not installed!')
    from ultralytics import YOLO

import ultralytics
ultralytics.checks()

try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f'error loading model: {e}')


# YOLO 모델 확인
print(model.names)  # COCO dataset : 80개의 클래스 존재
print(len(model.names)) # 80
"""
80개의 클래스 중 일부 클래스(ex 5개)만 분류하는 모델을 만들고 싶을 경우, 전이학습을 사용하면 됨.
"""

# dog 이미지 로딩 후 이미지 객체를 감지.
from PIL import Image               #Python image library, 파이썬 이미지 라이프러리(size, show, save, resize, crop, rotate ...)
import matplotlib.pyplot as plt

image_path = 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/dog.jpg'

try:
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f'error! {e}')
    exit()

import cv2  # 컴퓨터 비전, 영상 처리, 머신러닝 영상 관련 기능 제공 라이브러리.
import numpy as np

try:
    results = model(image)
except Exception as e:
    print(f'error during inference : {e}')
    exit()
#print(results)  # results.boxes, probs, names, plot, save, show
print(results[0].orig_shape)

# Pillow -> numpy 배열로 변환
image = np.array(image)
print(image.shape)          # (183, 275, 3)
print(image[:2, :2])
print(image[0, 0])          # [70 61 30]

cropped = image[:100, :100]
print('cropped : ', cropped)    #원래 이미지 중 좌 상단 100x100만 자름.
plt.imshow(cropped)
plt.axis('off')
plt.show()

# 감지된 객체 이미지에 박스 채우기
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for result in results:      #한장의 이미지씩 꺼내보기
    try:
        for box in result.boxes:    #바운딩 박스의 정보 리스트를 갖고 있음.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(x1, y1, x2, y2)   # 11 20 132 153
            label = result.names[int(box.cls[0])]       #tensor([16.])
            print(label)            # result.names의 16번째 값 : dog
            confidence = box.conf[0].item()     #신뢰도 수치 저장
            print('confidence : ', confidence)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} : {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        print(f'arr : processing error {e}')

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 감지 이미지를 저장
#cv2.imwrite('outTest1.jpg', image)
