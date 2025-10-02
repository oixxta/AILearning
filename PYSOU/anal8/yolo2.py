"""
YOLO

컴퓨터 카메라(웹캠)로 감지되는 물건 식별하기
Computer Vision(opencv:Open Source Computer Vision)
"""
import cv2
from ultralytics import YOLO
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO('yolov8n.pt')      # YOLO 버전8 나노 가져오기
#print(model.names)


# 감지된 이미지들을 저장할 폴더 생성
save_dir = 'C:/Users/msi/Desktop/Jongseong KIM/AILearning/PYSOU/anal8/yolo2dir'
os.makedirs(save_dir, exist_ok=True)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Webcam is not avaliable')
    exit()
else:
    print('Webcam is avaliable')
cv2.namedWindow('YOLO 실시간 객체 감지', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO 실시간 객체 감지', 800, 600)

# 중복저장 방지 : 3초 내에는 같은 객체 저장을 하지 않게.
last_save_time = {}

while True:
    ret, frame = cap.read()  # ret: 프레임 읽기 성패여부(TRUE or FALSE), frame: 읽은 프레임
    if not ret:
        print('프레임을 읽을 수 없음')
        break
    
    results = model(frame, verbose=False)   # model.predict(기본값 변경)

    # 특정 객체만 감지에 참여를 시킴.
    allowed_labels = ['person', 'laptop', 'keyboard', 'cell phone', 'book', 'clock']

    for result in results:
        for box in result.boxes:
            #특정 객체만 감지
            label = result.names[int(box.cls[0])]
            #if label != 'person': continue             #person만 허용
            if label not in allowed_labels : continue   #allowed_labels 이외의 것들 무시

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    #바운딩 박스 그리기
            cv2.putText(frame, f'{label} : {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #중복방지를 위한 2초 간격으로 저장
            now = time.time()    #현재시간 불러오기
            lastTime = last_save_time.get(label, 0)    #어떤 객체가 처음으로 감지되면 값이 없으니 0을 반환함.

            if now - lastTime >= 3:
                fileName = f'{label}_{int(now)}.jpg'    #저장할 파일 이름, 레이블 + 시간
                filePath = os.path.join(save_dir, fileName)
                cv2.imwrite(fileName, filePath)
                print(f'image saved : {filePath}')
                last_save_time[label] = now
        
    # 감지된 프레임 화면에 출력하기
    cv2.imshow('YOLO 객체 실시간 감지', frame)

    # q키를 누를 경우 웹캠 끄기
    key = cv2.waitKey(1)    # 1ms 동안 입력 대기하기. 아무키도 안 누르면 -1 반환
    if key != -1:
        print('눌린 키 : ', key, chr(key))
    #print('눌린 키 : ', key)
    if key &0xFF == ord('q'):
        break

# 자원 정리(웹캠)
cap.release()       #사용중인 카메라 장치 해제
cv2.destroyAllWindows()     #openCv가 만든 모든 창을 닫음.
