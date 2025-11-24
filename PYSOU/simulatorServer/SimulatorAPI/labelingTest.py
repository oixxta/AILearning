# auto_detect_all_obstacles_filter.py
# ---------------------------------------------
# YOLOv9e로 폴더 내 이미지 모든 객체 탐지
# 탐지된 객체 1개만 시각화 + 라벨링
# 0개 또는 2개 이상 탐지 시 원본 삭제
# ---------------------------------------------

import os
import cv2
from ultralytics import YOLO
import shutil

# =============== 설정 ===============
INPUT_DIR = r"c:\PYSOU\final_project\labeling\input"        # 이미지 폴더
OUTPUT_DIR = r"c:\PYSOU\final_project\labeling\output_filtered"  # 시각화 결과 저장 폴더
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")              # 라벨 파일 저장 폴더
CONF_THRES = 0.5  # confidence threshold

classification_count = 0

# YOLOv9e 모델 불러오기 (pretrained)
model = YOLO("yolov9e.pt")

# =============== 준비 단계 ===============
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# =============== 탐지 및 시각화 저장 ===============
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"[INFO] 총 {len(image_files)}개의 이미지 탐지 시작...")

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # YOLOv9e 예측
    results = model.predict(source=img_path, conf=CONF_THRES, verbose=False)

    # 모든 객체 수집
    detected_boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detected_boxes.append((cls_id, cls_name, box, conf))

    # 1개만 감지된 경우 저장
    if len(detected_boxes) == 1:
        classification_count += 1
        cls_id, cls_name, box, conf = detected_boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = ((x1 + x2)/2)/w
        y_center = ((y1 + y2)/2)/h
        box_w = (x2 - x1)/w
        box_h = (y2 - y1)/h

        # 라벨 파일 저장
        label_file = os.path.join(LABELS_DIR, img_name.rsplit('.', 1)[0] + ".txt")
        with open(label_file, "w") as f:
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # 시각화
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 시각화 이미지 저장
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, img)
        print(f"[SAVED] {img_name} ({cls_name})")

    #else:
    #    # 0개 또는 2개 이상이면 삭제
    #    os.remove(img_path)
    #    print(f"[DELETED] {img_name} (탐지 객체 수: {len(detected_boxes)})")

print(f"\n 완료! 시각화 이미지: '{OUTPUT_DIR}/'")
print(f"   라벨 파일: '{LABELS_DIR}/'")
print(f"분류 성공 이미지 갯수 : {classification_count}")
