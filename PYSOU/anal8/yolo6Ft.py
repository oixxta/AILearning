"""
YOLO 탐지 결과를 이미지별로 정리해서 CSV 파일로 저장해 내보내기
이어서 CSV를 읽어 DataFrame애 담아 다른 것들을 해보기.

그리고 파인튜닝으로 모델의 일부를 수정
"""
import os
import pandas as pd
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
img_dir = "anal8\images"
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(img_paths)

records = []    #결과를 담을 리스트형 변수

for path in img_paths:
    results = model(path, conf=0.25, verbose=False)[0]
    boxes = results.boxes
    names = results.names
    #print(boxes, names)     #3개의 텐서가 출력됨.
    if len(boxes) == 0:
        records.append({
            'image' : os.path.basename(path),
            'object_count' : 0,
            'classes' : '',
            'avg_confidence' : 0.0,
        })
        continue

    class_ids = boxes.cls.cpu().numpy().astype(int)
    print(class_ids)
    confs = boxes.conf.cpu().numpy()
    print(confs)
    classes = [names[i] for i in class_ids]
    print(classes)
    avg_conf = float(confs.mean())

    records.append({
        'image' : os.path.basename(path),
        'object_count' : len(class_ids),
        'classes' : ','.join(sorted(set(classes))),
        'avg_confidence' : round(avg_conf, 3)
    })

# records -> Dataframe -> csv
dataframe = pd.DataFrame(records)
print(dataframe)
"""
        image  object_count     classes  avg_confidence
    0  image1.jpg             1      person           0.866
    1  image2.jpg             5      person           0.547
    2  image3.jpg            15  dog,person           0.653
"""
dataframe.to_csv('yotest6report.csv', index=False, encoding='utf-8-sig')
print('CSV 저장 완료')


# 위에 저장한 CSV를 다시 불러와 또다른 데이터프레임으로 만들기
myDf = pd.read_csv('yotest6report.csv')
num_images = len(myDf)
total_objects = myDf['avg_confidence'].sum()
print('total_objects', total_objects)   #총 탐지갯수

# 전체 신뢰도 평균 구하기
overall_avg_conf = myDf.loc[myDf['avg_confidence'] > 0, 'avg_confidence'].mean() if total_objects > 0 else 0.0

# 클래스별 등장 빈도
class_counts = {}
for cls_str in myDf['classes']:
    if cls_str:
        for c in cls_str.split(','):
            class_counts[c] = class_counts.get(c, 0) + 1


# 최종 감지보고서
print('== YOLO Detection Summary ==')
print('total image count :           ', num_images)
print('total detected object count : ', total_objects)
print('total confidence average :    ', overall_avg_conf)
print('type count each classes :     ')
for k, v in class_counts.items():
    print(f'   {k} : {v}')