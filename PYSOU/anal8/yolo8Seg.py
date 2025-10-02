"""
이미지 세그멘테이션(Image Segmentation)

Image Detection(객체 검출) 과 Image Segmentation(이미지 분할) 
은 둘 다 컴퓨터 비전의 대표적인 처리 형태지만, 출력의 형태와 세밀함 수준에서 큰 차이가 있다.

Image Detection(객체 검출) 과 Image Segmentation(이미지 분할) 비교용 코드 작성 및 연습
"""
import os, cv2, numpy as np
from ultralytics import YOLO

IMG_PATH = 'image1.jpg'
OUT_DIR = 'seg_out'
os.makedirs(OUT_DIR, exist_ok=True)

im = cv2.imread(IMG_PATH)
assert im is not None, f'이미지 읽기 실패 : {IMG_PATH}'

H, W = im.shape[:2]
print(H, W) # (1635, 1020, 3), 원본 이미지 크기는 마스크 리사이징때 필요


# 모델 생성하기
#det_model = YOLO("yolov8n.pt")       # detection 전용
model = YOLO("yolov8n-seg.pt")        # segmentation 전용

res = model(im)[0]
#print(res)              # boxes, masks, names, array 등을 제공함.

cv2.imwrite(os.path.join(OUT_DIR, 'anno1.jpg'), res.plot())
#res.plot() : 원본 이미지 위에 바운딩박스, 레이블, 신뢰도 점수, 세그멘테이션 마스크를 한번에 그려서 BGR 이미지로 제공함.

#마스크가 없을 경우, 작업을 종료함.
if res.masks is None or len(res.masks.data) == 0:
    print('마스크 없엉')
    raise SystemExit

m_small = res.masks.data.cpu().numpy()
#print(m_small)
masks = np.stack([
    cv2.resize(m, (W, H), cv2.INTER_NEAREST) > 0.5 for m in m_small
], axis=0)  #각 객체별 (H, W) 마스크를 모아서 (N, H, W) 배열로 만듬. N개의 bool 마스크 스택
#print(masks)

# 세그멘테이션 전 단계 : 마스크 프리뷰
# 마스크가 같은 위치 픽셀에 대해 객체 중 하나라도 1(TRUE)이면, N개 마스크를 OR 연산으로 합침.
mask_union = (masks.any(axis=0).astype(np.uint8) * 255)
cv2.imwrite(os.path.join(OUT_DIR, 'mask_preview.jpg'), mask_union)


# 최종 세그멘테이션 : 컬러 오버레이 + 하얀색 외곽선
def color(i):
    return ((37 * i) % 256, (17 * i) % 256, (91 * i) % 256)
final = im.copy()           #직접 원본에 덧칠이 아닌, 합성용 이미지 캔버스 복사
blend = np.zeros_like(im)   #오버레이 색 채우기 캔버스.

for i, m in enumerate(masks):    #컬러 오버레이 작업(blend)는 객체 내부를 채색 및 외곽선 생성
    blend[m] = color(i)          #마스크 영역에 칠해질 고유 색상 채우기
    cnts, _ = cv2.findContours(  #마스크의 윤곽선 좌표를 추출함.
        (m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # 0/1 -> 0 ~ 255 사이로 이진화      가장 바깥쪽 외곽선만 꼭짓점 단순화
    )
    cv2.drawContours(final, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)

# 반투명 합성
final = cv2.addWeighted(final, 1.0, blend, 0.45, 0.0)
cv2.imwrite(os.path.join(OUT_DIR, 'final_preview.jpg'), final)

#cv2.imshow('final segmentation', final)
#cv2.waitKey(0)
#cv2.destroyAllWindows()