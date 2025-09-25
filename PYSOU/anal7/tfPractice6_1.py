"""
tPractice6에서 만든 모델파일 시험용
"""
import os, random, pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar100

# ----------------------------
# 0) CIFAR-100 클래스 이름 (fine labels)
# ----------------------------
CIFAR100_FINE_LABELS = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup',
    'dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house',
    'kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',
    'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid',
    'otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy',
    'porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal',
    'shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar',
    'sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# (윈도우 한글 폰트가 필요하면 주석 해제)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 1) 모델 불러오기
# ----------------------------
MODEL_PATHS = [
    'tfPractice6_CIFAR-100.keras',   # 최종 저장 파일
    'mnv2_cifar100.keras'            # 콜백 체크포인트(최고 성능)
]

model_path = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError("모델 파일을 찾을 수 없습니다. 'tfPractice6_CIFAR-100.keras' 또는 'mnv2_cifar100.keras'가 현재 폴더에 있는지 확인하세요.")

print(f"[INFO] Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# ----------------------------
# 2) 데이터 불러오기 & 전처리 함수 (훈련과 동일)
# ----------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 64
NUM_CLASSES = 100

(xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='fine')

def preprocess(x, y):
    x = tf.image.resize(x, IMG_SIZE)  # uint8 -> float32
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1, 1]
    y = tf.squeeze(y, axis=-1)        # (N,1) -> (N,)
    return x, y

testDs = (
    tf.data.Dataset
    .from_tensor_slices((xTest, yTest))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------
# 3) 테스트셋 평가
# ----------------------------
loss, acc = model.evaluate(testDs, verbose=1)
print(f"[RESULT] Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# ----------------------------
# 4) 샘플 N개 예측 시각화 (테스트셋에서)
# ----------------------------
def predict_topk(images, k=5):
    """모델 확률 예측 -> top-k (index, prob) 반환"""
    probs = model.predict(images, verbose=0)  # (N, 100)
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    topk_prob = np.take_along_axis(probs, topk_idx, axis=1)
    return topk_idx, topk_prob

N = 8  # 그려볼 샘플 수
indices = np.random.choice(len(xTest), size=N, replace=False)
images = xTest[indices]                            # (N, 32, 32, 3)
labels = yTest[indices].reshape(-1)               # (N,)

# 전처리 후 예측
images_tf = tf.convert_to_tensor(images)
images_tf = tf.image.resize(images_tf, IMG_SIZE)
images_tf = tf.keras.applications.mobilenet_v2.preprocess_input(images_tf)
topk_idx, topk_prob = predict_topk(images_tf)

# 시각화
cols = 4
rows = int(np.ceil(N / cols))
plt.figure(figsize=(4 * cols, 3.8 * rows))

for i in range(N):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    gt_idx = int(labels[i])
    gt_name = CIFAR100_FINE_LABELS[gt_idx]
    preds = [(CIFAR100_FINE_LABELS[int(topk_idx[i, j])], float(topk_prob[i, j]))
             for j in range(topk_idx.shape[1])]
    title_lines = [f"GT: {gt_name}"]
    title_lines += [f"{j+1}) {name} ({p*100:.1f}%)" for j, (name, p) in enumerate(preds)]
    plt.title("\n".join(title_lines), fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# ----------------------------
# 5) 임의의 로컬 이미지 1장 추론
#    - CIFAR-100 클래스 중 무엇처럼 보이는지 top-5 출력
# ----------------------------
def predict_file(img_path, show=True):
    """
    img_path: 이미지 파일 경로 (RGB 이미지 권장)
    """
    import PIL.Image as Image
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)  # H, W, 3

    # 전처리
    x = tf.image.resize(img_np, IMG_SIZE)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = tf.expand_dims(x, axis=0)  # (1, H, W, 3)

    # 예측
    probs = model.predict(x, verbose=0)[0]  # (100,)
    top5 = np.argsort(-probs)[:5]
    result = [(CIFAR100_FINE_LABELS[i], float(probs[i])) for i in top5]

    if show:
        plt.figure(figsize=(4,4))
        plt.imshow(img_np)
        plt.axis('off')
        title = "Top-5 예측:\n" + "\n".join([f"{k+1}) {name} ({p*100:.1f}%)" for k,(name,p) in enumerate(result)])
        plt.title(title, fontsize=10)
        plt.show()

    return result

result = predict_file("sample.jpg", show=True)
print(result)
