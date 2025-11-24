"""
vecdb4를 응용
VectorDB에 이미지 자료를 IO
이미지 벡터화 처리는 Resnet

"""
import os
import torch
from torchvision import transforms
from PIL import Image
from chromadb import PersistentClient
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import koreanize_matplotlib


### 이미지를 이미지 벡터화 시키기(임베딩, resnet18을 이용한 임베딩)
model = resnet18(weights='ResNet18_Weights.DEFAULT')
model.eval()    # 모델은 평가 모드로 전환함.(훈련 X, 추론처럼 동작함.)
model = torch.nn.Sequential(*(list(model.children())[:-1]))   # 이미 학습된 모델의 구성 레이어를 리스트화, 마지막 것(FC층, 분류)은 제거.


### FC를 제외한 나머지 레이어들을 순차적으로 묶은 새로운 모델로 재구성
# 이미지를 불러 ResNet18을 이용해 512차원 벡터로 반환하는 함수
def image_to_vectorFunc(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # ResNet18은 입력 이미지 크기 (3, 224, 224)를 원함.
        transforms.ToTensor(),              # PIL.image를 torch.tensor로 변환함.
    ])
    tensor = transform(image).unsqueeze(0)  # 모델에 넣기 위해 배치차원 앞에 1을 추가함.((1, 3, 224, 224))
    with torch.no_grad():                   # 역전파 계산을 끄고, 추론만 사용.
        vec = model(tensor).squeeze().numpy()   #(512,) 모든 크기의 1인 차원을 제거함.
        # vec의 결과는 (1, 512, 1, 1) -> (512,)
    print(f'{img_path} -> 벡터(앞 10개): {vec[:10]}')
    return vec.tolist()                     # 제이슨 직렬화를 위해 넘파이 대신 리스트로 리턴.

### 벡터에 저장하기
client = PersistentClient('./image_chroma2')
collection = client.get_or_create_collection(name='images')

### 이미지 경로 잡기
file_names = ['apple.jpg', 'banana.jpg', 'peach.jpg']
image_files = [os.path.join('C:/Users/msi/OneDrive/바탕 화면/Jongseong KIM/AILearning/PYSOU/anal11/pic', name) for name in file_names]
print(image_files)
image_ids = [f'img{i}' for i in range(len(image_files))]
print('ids:', image_ids)

for img_id, img_path in zip(image_ids, image_files):
    if not os.path.exists(img_path):
        print(f'파일 없음 : {img_path}')
        continue
    vec = image_to_vectorFunc(img_path)
    collection.add(
        embeddings=[vec],
        documents=[img_path],
        ids=[img_id],
        metadatas=[{'filename' : img_path}]
    )
