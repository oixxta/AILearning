"""
Tensorflow 기반 이미지를 VectorDB에 IO
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from chromadb import PersistentClient
import os



# ResNet50 모델 로딩 (분류기 제거)
model = tf.keras.applications.ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    input_shape=(224, 224, 3)
)

def image_to_vector_tf(img_path):
    image = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
    vector = model.predict(img_array)[0]
    print(vector)
    return vector.tolist()

# Chroma
Client = PersistentClient(path='./image_chroma_tf')
collection = Client.get_or_create_collection(name='images_tf')

image_files = ['apple.jpg', 'banana.jpg', 'peach.jpg']
ids = [f'img{i}' for i in range(len(image_files))]
print(ids)

for img_id, img_path in zip(ids, image_files):
    if not os.path.exists(img_path):
        print(f'파일 없음 : {img_path}')
        continue
    vec = image_to_vector_tf(img_path)

# 쿼리 이미지 검색
query_path = 'apple.jpg'
if not os.path.exists(img_path):
    print(f'파일 없음 : {img_path}')
else:
    query_vec = image_to_vector_tf(query_path)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        include=['documents', 'distances', 'metadatas']
    )
    print('검색 결과 : ')
    for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        print(f' - {meta['filename']} (유사도 거리 : {dist:.4f})')




