"""
YOLO

로컬 내 파일을 읽어와서 구분시키기
"""
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt


# 모델 가져오기
model = YOLO('yolov8n.pt')

# 작업할 이미지 가져오기
image_path = 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/anal8/yolo3dir/pic2.jpg'
try:
    image = cv2.imread(image_path)
except FileNotFoundError as e:
    print('이미지를 읽어오지 못함.')
    raise SystemExit

original = image.copy()
results = model(image)
print(results)

dogCount = 0        #감지된 물체의 수를 저장시킬 변수

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        if label.lower() == 'person':
            dogCount += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)    #바운딩 박스 그리기
        cv2.putText(image, f'{label} : {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f'감지된 개의 수 : {dogCount} 마리')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'detected person : {dogCount}')
plt.show()

# 바운딩 박스가 된 이미지 전체를 저장하기
#outPath = 'yotest3_out.jpg'
#cv2.imwrite(outPath, image)
#print("바운딩 박스가 적용된 이미지 저장 완료")


# 바운딩 박스 내부 객체만 저장(박스선 제거 + 세그멘테이션 적용)
for idx, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        # 원본 이미지에서 ROI(region of interest, 관심영역) 추출하기
        cropped = original[y1:y2, x1:x2]   # image(H, W, 3)를 배열 슬라이싱으로 선택된 영역만 선택
        print('cropped : ', cropped)

        #선택된 이미지 배열 파일로 저장
        #cropPath = f'crop_{idx}_{j}_{label}_{confidence:.2f}.jpg'
        #cv2.imwrite(cropPath, cropped)
        #print(f'객체 {label}이 성공적으로 저장됨.')


#감지된 객체의 중심점 좌표 출력하기
dCount = 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = result.names[int(box.cls[0])]
        confidence = box.conf[0].item()

        centerX = (x1 + x2) // 2
        centerY = (y1 + y2) // 2

        if label.lower() == 'person':
            dCount += 1 
            print(f'person => {dCount} : 중심좌표는 {centerX}, {centerY}, 신뢰도:{confidence:.2f}')
            # 중심점 그리기
            cv2.circle(image, (centerX, centerY), 5, (0, 0, 255), -1)

            coordText = f'({centerX},{centerY})'
            cv2.putText(image, coordText, (centerX + 10, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()