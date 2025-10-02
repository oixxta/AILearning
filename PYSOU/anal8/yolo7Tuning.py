"""
YOLO8 모델을 파인튜닝 해서 다른 이미지 레이블 감지 모델 만들기

로보플로우에서 해산물 분류 모델을 사용.
사용 데이터 : https://public.roboflow.com/object-detection/aquarium/2/download/yolov8
데이터 폴더 : Aquarium_Data
"""
import yaml
from IPython.display import display
import ultralytics
from ultralytics import YOLO
import numpy as np

# 추가 데이터 가져오기
data = {
    #데이터 폴더의 data.ymal 파일
    'train': 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/Aquarium_Data/train/images',
    'validation': 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/Aquarium_Data/valid/images',
    'test': 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/Aquarium_Data/test/images',
    'names': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray'],
    'nc': 7             #클래스 수
}

with open('Aquarium_Data.yaml', 'w') as f:
    yaml.dump(data, f)                                          #위에서 정의한 데이터를 Aquarium_Data.yaml로 새로 저장

with open('Aquarium_Data.yaml', 'r') as f:
    aquarium_yaml = yaml.safe_load(f)
    display(aquarium_yaml)


# YOLO 모듈 호출
model = YOLO('yolov8n.pt')
print(type(model.names), len(model.names))
print(model.names)

model.train(data='Aquarium_Data.yaml', epochs=100, patience=30, batch=32, imgsz=416, device=0, workers=0, amp=True)    #YOLO 계열 최적화 이미지 사이즈 : 416, 512, 640
print(type(model.names))

# test 이미지 생성 및 확인
from glob import glob   # 와일드카드 맵핑용 *, ?

test_image_list = glob('C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/Aquarium_Data/test/images/*')
print(test_image_list)

test_image_list.sort()
for i in range(len(test_image_list)):
    print('i=', 1, test_image_list[i])

# 예측
results = model.predict(source='C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/Aquarium_Data/test/images/*', save=True)
print(type(results), len(results))

# 클래스별 검출 결과 확인하기


for result in results:
    uniq, cnt = np.unique(result.boxes.cls.cpu().numpy(), return_counts=True)
    uniq_cnt_dict = dict(zip(uniq, cnt))    # 두 배열을 묶어서 dic 형태로
    print(uniq_cnt_dict)
    for c in result.boxes.cls:
        print('class_num = ', int(c), ' ', 'class_name=', model.names[int(c)])

# 예측된 이미지 파일 목록 확인
#detected_image_list = glob.glob((''))


# 파인튜닝된 새로운 모델로 새로운 이미지에 대한 객체 검출
myModel = YOLO('C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/runs/detect/train/weights/best.pt')
print(myModel)
image_path = 'shark.jpg'
results = myModel.predict(source=image_path, save=True, imgsz=416)

# 결과 읽기
from pathlib import Path
result_img_path = Path(results[0].save_dir) / Path(image_path).name
print(result_img_path)

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open(result_img_path)
plt.imshow(img)
plt.axis('off')
plt.show()


# 탐지 객체 정보
from collections import defaultdict
detected_class = []
conf_dict = defaultdict(list)

for box in results[0].boxes:
    cls_id = int(box.cls)
    cls_name = myModel.names[cls_id]
    conf = float(box.conf)
    detected_class.append(cls_name)
    conf_dict[cls_name].append(conf)

print('탐지된 클래스 전체 : ', detected_class)
print('고유 클래스 : ', sorted(set(detected_class)))

for cls_name, confs in conf_dict.items():
    print(f' - {cls_name} : 갯수 = {len(confs):.2f}, 평균신뢰도={np.mean(confs):.3f}')
