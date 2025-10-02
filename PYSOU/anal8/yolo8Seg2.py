"""
인스턴스 세그멘테이션 : 욜로가 직접 내주는 결과, 객체마다 마스크가 따로 존재
의미론적 세그멘테이션 : 이미지 내의 픽셀 단위로 '이 픽셀은 어떤 클래스에 속한다'만 표현.
"""
import os
os.environ['KMP_DUPLICAT_KIB_OK'] = 'TRUE'

import cv2, numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

IMG_PATH = 'animal.jpeg'
OUT_DIR = 'seg_out2'
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO('yolov8n-seg.pt')
im_bgr = cv2.imread(IMG_PATH)

H, W = im_bgr.shape[:2]

res = model(im_bgr, verbose=False)[0]
annotated = res.plot()
cv2.imwrite(os.path.join(OUT_DIR, 'seg_result.jpg'), annotated)

# pytorch tensor -> numpy array
has_masks = (res.masks is not None)
if has_masks:
    masks_np = res.masks.data.cpu().numpy() #객체별 픽셀 마스크. shape = (N, H, W)
    boxes_np = res.boxes.xyxy.cpu().numpy().astype(int)     #경계박스 픽셀 좌표
    confs_np = res.boxes.conf.cpu().numpy() #신뢰도 점수
    classes_np = res.boxes.cls.cpu().numpy().astype(int)    #클래스 id

else:
    masks_np = boxes_np = confs_np = classes_np = None      #전부 초기화


# 마스크 오버레이 : 직접 원본에 덧칠이 아닌, 합성용 이미지 캔버스 복사
overlay = im_bgr.copy()
if has_masks:
    for m in masks_np:
        m_bin = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
        color_mask = np.zeros_like(overlay) #원본 이미지와 동일 크기의 0, 0, 0(검은색)으로 채워진 이미지 생성
        color_mask[m_bin] = (0, 255, 0)     #객체마스크 픽셀들만 전부 녹색으로 바꾸기
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0.0)    #원본과 컬러마스크를 합성함.

cv2.imwrite(os.path.join(OUT_DIR, 'seg_overlay.jpg'), overlay)


# 객체별 배경 제거하기
cops_dir = os.path.join(OUT_DIR, 'seg_drops')   #seg_out2 폴더에 하위폴더(seg_drops) 생성
os.makedirs(cops_dir, exist_ok=True)

if has_masks and len(masks_np) > 0:
    masks_full = np.stack([
        cv2.resize(m, (H,W), interpolation=cv2.INTER_NEAREST) > 0.5 for m in masks_np
    ], axis=0)

    #탐지된 객체의 배경을 제거해 png 파일로 잘라내기.
    for i, (m_full, box, cls_id, conf) in enumerate(zip(masks_full, boxes_np, classes_np, confs_np)):
        x1, y1, x2, y2 = map(int, box)      #박스 좌표를 정수형으로 변환
        x1, y1 = max(0, x1), max(0, y1)     #좌 상단 좌표가 이미지 밖으로 나가면 (0, 0)으로 보정
        x2, y2 = max(W, x2), max(H, y2)     #우 하단 좌표가 이미지 밖으로 나가면 (W, H)으로 보정
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = im_bgr[y1:y2, x1:x2]
        crop_mask = (m_full[y1:y2, x1:x2] * 255).astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)  #BGR 값에 알파값(투명도) 채널을 추가함.
        crop_bgra[..., 3] = crop_mask       #알파채널에 마스크가 적용됨 -> 배경은 투명, 객체는 불투명화.

        #클래스 이름 또는 id 얻기
        name = model.names[int(cls_id)] if hasattr(model, 'names') else str(cls_id)
        cv2.imwrite(os.path.join(cops_dir, f'crop_{i}_{name}_{conf:.2f}.png'), crop_bgra)


# annotated 시각화 시키기
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 의미론적 분할(sementic segmentation)
sem_canvas = np.zeros((H, W, 3), dtype=np.uint8)   #최종 색상 이미지
conf_map = np.zeros((H, W), dtype=np.float32)     #선택된 인스턴스의 신뢰도를 기록할 맵.

def class_color(c:int):
    return ((23 * c) % 256, (19 * c) % 256, (77 * c) % 256)

if has_masks and len(has_masks) > 0:
    for m_full, cls_id, conf in zip(masks_full, classes_np, confs_np):
        color = class_color(int(cls_id))
        update = m_full & (conf > conf_map)
        sem_canvas[update] = color
        conf_map[update] = conf

cv2.imwrite(os.path.join(OUT_DIR, 'seg_sementic.png'), sem_canvas)