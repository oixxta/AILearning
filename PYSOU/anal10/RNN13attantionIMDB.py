# IMDB BiLSTM + Additive(Bahdanau) Attention
# Bahdanau 어텐션의 원리를 반영한 AdditiveAttention(가중합 기반)을 사용
# - 어텐션 출력은 항상 3D로 만든 뒤 GlobalAveragePooling1D로 2D로 축소(안전)

import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.layers import (Input, Embedding, Bidirectional, LSTM,
                                     Dense, Dropout, Concatenate, AdditiveAttention, Reshape, GlobalAveragePooling1D)
from keras.models import Model

# 1) 데이터 로드 & 패딩
# IMDB 데이터셋은 영화 리뷰(텍스트)와 감성(긍정=1, 부정=0)으로 구성
# num_words=10000 → 상위 10,000개의 단어만 사용
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# 리뷰 길이는 제각각이므로 패딩(padding) 필요. 오른쪽에 0을 채우는 'post' 패딩을 사용
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
X_test  = pad_sequences(X_test,  maxlen=max_len, padding='post', truncating='post')

# 2) 모델 정의
# 입력층 (영화 리뷰 시퀀스: 단어 인덱스 시퀀스)
seq_in = Input(shape=(max_len,), dtype="int32")

# 임베딩층
# - 단어 인덱스를 128차원 밀집벡터로 변환
# - mask_zero=True → 0(패딩 토큰)을 무시하도록 마스크 생성
emb = Embedding(vocab_size, 128, mask_zero=True)(seq_in)

# 첫 번째 BiLSTM 층
# - 양방향 LSTM (순방향 + 역방향)
# - return_sequences=True → 모든 시점의 은닉상태 출력
# - dropout=0.5 → 과적합 방지
x = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(emb)

# 두 번째 BiLSTM 층 (상태 반환)
# - return_sequences=True → 모든 시점의 출력 유지 (어텐션 입력으로 사용)
# - return_state=True → 마지막 시점의 상태도 함께 반환
#   forward_h, backward_h: 각 방향의 마지막 은닉 상태
#   forward_c, backward_c: 각 방향의 마지막 셀 상태
x, fh, fc, bh, bc = Bidirectional(
    LSTM(64, dropout=0.5, return_sequences=True, return_state=True)
)(x)

# 은닉 상태 결합
# - 순방향과 역방향의 마지막 은닉 상태를 연결(concatenate)
# - 즉, 전체 문장의 요약 정보를 담은 쿼리(query) 벡터
state_h = Concatenate()([fh, bh])          # (batch_size, 128)

# 쿼리(query)를 3D로 변환
# AdditiveAttention은 (batch, time, hidden) 3D 입력을 기대하므로
# 쿼리의 차원을 (batch, 1, hidden_size)로 확장
query3d = Reshape((1, 128))(state_h)       # (B, 1, 128)

# 마스크 설정
# Embedding에서 자동 생성된 마스크를 어텐션에 전달
# → 패딩 토큰(0)은 주의(attention)를 받지 않도록 함
value_mask = emb._keras_mask               # (B, T) boolean

# 어텐션 레이어
# AdditiveAttention = Bahdanau Attention과 동일한 additive score 사용:
#   score = v^T * tanh(W1 * value + W2 * query)
# query : 문장 요약(hidden state)
# value : 모든 시점의 LSTM 은닉 상태
attn = AdditiveAttention()

# AdditiveAttention의 입력
#   [query, value]  +  [mask_query, mask_value]
# query_mask는 None, value_mask는 Embedding에서 전달받은 마스크
context3d = attn([query3d, x], [None, value_mask])   # (B, 1, 128)

# 컨텍스트 벡터 축소
# AdditiveAttention 출력은 (B, 1, 128)
# GlobalAveragePooling1D()을 이용해 시퀀스 차원(길이 1)을 제거하고 (B, 128)로 변환
context = GlobalAveragePooling1D()(context3d)

# 분류 헤드 (Fully Connected Layer)
# context → Dense → Dropout → Sigmoid
# ReLU 활성화로 비선형성 추가
h = Dense(20, activation="relu")(context)
h = Dropout(0.5)(h)
out = Dense(1, activation="sigmoid")(h)    # 이진 분류 (0=부정, 1=긍정)

# 모델 구성 및 컴파일
model = Model(seq_in, out)
model.compile(
    optimizer="adam",                # Adam 옵티마이저 (학습 안정적)
    loss="binary_crossentropy",      # 이진 감성 분류에 적합한 손실 함수
    metrics=["accuracy"]             # 정확도 모니터링
)
model.summary()

# 3) 학습
# - 3 epoch, batch=256으로 학습
# - 검증 데이터로 X_test 사용
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# - 테스트 정확도 계산
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("\n테스트 정확도: %.4f" % test_acc)

# 1) 테스트 세트 일부 예측
import numpy as np

# 학습된 model로 테스트 샘플 10개에 대해 확률 예측 수행
# model의 마지막 층이 sigmoid(1 유닛)이므로 출력 형태는 (배치, 1)
probs = model.predict(X_test[:10], batch_size=256)   # 예: (10, 1)

# 확률을 0.5 임계값으로 이진 라벨로 변환
# probs.ravel() → (10,)으로 평탄화; 0=부정, 1=긍정
preds = (probs.ravel() >= 0.5).astype(int)
print("예측 확률:", np.round(probs.ravel(), 4)) # 예측 확률을 소수점 4자리로 출력
print("예측 라벨:", preds.tolist())   # 예측 라벨(0/1) 리스트로 출력

# 같은 위치의 정답 라벨과 비교할 수 있도록 실제 라벨 출력
print("실제 라벨:", y_test[:10].tolist())


# 2) 임의 문장(raw text) → 감성 예측 함수
import re  # 간단한 텍스트 정규식 전처리용

# IMDB의 단어 사전 로드
# 기본 반환은 {'the': 1, 'and': 2, ...} 형태의 원본 인덱스(0,1,2는 예약된 토큰을 위해 비워둠)
word_index = imdb.get_word_index()  # 원본 단어→인덱스 매핑

index_from = 3     # 0,1,2는 각각 PAD/START/OOV로 예약 → 실제 단어는 3부터 시작
# 사전의 모든 인덱스를 +3 만큼 시프트하여 Keras IMDB 규약과 맞춤
word_index = {w: (i + index_from) for w, i in word_index.items()}

# 모델이 학습 시 사용한 특수 토큰 정의
PAD = 0          # pad_sequences가 채우는 값(패딩)
START = 1        # 문장 시작 토큰
OOV = 2          # 사전에 없는(out-of-vocabulary) 단어에 부여할 토큰
# 주의: 실제 단어 인덱싱은 3부터 시작(index_from)

def text_to_sequence(text: str):
    """
    임의의 영어 문장을 IMDB 인덱스 시퀀스로 변환.
    - 소문자화 및 간단 정규식으로 알파벳/숫자/'/'공백만 남김
    - START 토큰을 맨 앞에 추가
    - 사전에 없는 단어는 OOV 토큰으로 대체
    - 학습과 동일하게 오른쪽 패딩/오른쪽 자르기 적용
    """
    text = text.lower()  # 대소문자 차이 제거
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # 영어/숫자/'/공백 외 제거
    tokens = text.split()  # 공백 기준 토큰화

    # 시작 토큰을 추가하고, 각 토큰을 사전 인덱스로 매핑(OOV 처리 포함)
    seq = [START] + [word_index.get(tok, OOV) for tok in tokens]

    # 학습 시 입력과 동일한 길이/방식으로 패딩: 오른쪽 패딩, 오른쪽 자르기
    seq = pad_sequences(
        [seq],                 # pad_sequences는 2D 입력을 받음 → [seq]
        maxlen=max_len,        # 모델 학습 시 사용한 고정 길이와 동일해야 함
        padding='post',        # 오른쪽에 PAD(0)로 채움
        truncating='post',     # 길이가 길면 오른쪽을 잘라냄
        value=PAD              # 패딩 토큰 값(0)
    )
    return seq  # 형태: (1, max_len)

def predict_sentiment(text: str, threshold: float = 0.5):
    """
    단일 문장 텍스트에 대해 감성 확률과 라벨을 반환.
    - threshold 이상이면 긍정(1), 미만이면 부정(0)
    """
    seq = text_to_sequence(text)                     # (1, max_len)
    prob = float(model.predict(seq, verbose=0)[0][0])  # sigmoid 확률 스칼라값
    label = 1 if prob >= threshold else 0              # 임계값 기준 이진 분류
    return prob, label

# 사용 예시 문장들
samples = [
    "This movie was absolutely fantastic! The performances were brilliant and I loved the ending.",
    "Terrible plot and awful acting. I regret wasting my time.",
    "Not bad overall, a bit slow in the middle but the last act was solid.",
]

# 각 문장에 대해 예측 확률과 라벨 출력
for s in samples:
    p, y = predict_sentiment(s)
    print(f"[{s}]\n -> prob={p:.4f}, pred={'POSITIVE' if y==1 else 'NEGATIVE'}\n")