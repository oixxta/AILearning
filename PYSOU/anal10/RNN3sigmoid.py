"""
텍스트 정제 후 감성분류

데이터 분석 알고리즘은 모든 데이터를 수치로 표현했을 때, 비로소 제대로 작업 가능.

"""
import string
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input, Flatten

# 단어사전을 만들 때 사용할 토큰 인덱스.
token_index = {}

# 샘플 데이터 생성
samples = ["The cat say on the mat.", "The dangdang ate\t my\n homework."]

# 샘플 데이터를 수치화 1 : 단어 사전 생성 - 프로그래밍
for i in samples:
    for word in i.split():
        word = word.strip(string.punctuation).lower()       # 모든 문자들을 소문자화, 구두점 제거 및 공백을 기준으로 자름.
        print(word)

        if word not in token_index:
            token_index[word] = len(token_index)   # 단어별로 인덱스 부여 : 단어사전 만들기.

print(token_index)  # {'the': 0, 'cat': 1, 'say': 2, 'on': 3, 'mat': 4, 'dangdang': 5, 'ate': 6, 'my': 7, 'homework': 8}


# 샘플 데이터를 수치화 2 : 단어 사전 생성 - Tokenizer
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(samples)
print(tokenizer.word_index)     # {'the': 1, 'cat': 2, 'say': 3, 'on': 4, 'mat': 5, 'dangdang': 6, 'ate': 7, 'my': 8, 'homework': 9}
token_seq = tokenizer.texts_to_sequences(samples)
print(token_seq)                # [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
token_mat = tokenizer.texts_to_matrix(samples, mode='binary')
print(token_mat)
# [[0. 1. 1. 1. 1. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 1. 1. 1. 1.]]
# 원-핫 인코딩은 아니지만, 0과 1로만 된 것으로 바꿈.
print(tokenizer.word_counts)        # OrderedDict([('the', 3), ('cat', 1), ('say', 1), ('on', 1), ('mat', 1), ('dangdang', 1), ('ate', 1), ('my', 1), ('homework', 1)])
print(tokenizer.document_count)     # 2 
print(tokenizer.word_docs)          # defaultdict(<class 'int'>, {'on': 1, 'mat': 1, 'cat': 1, 'say': 1, 'the': 2, 'dangdang': 1, 'ate': 1, 'homework': 1, 'my': 1})

seq = token_seq[0]
num_classes = max(seq) + 1
print(num_classes)
token_seq = to_categorical(token_seq[0], num_classes=num_classes)   # 원 핫 인코딩 처리.
print(token_seq)
# [[0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1.]]


print('영화 관람 후 평에 대한 선호 분류')
docs = ['너무 재밋세여', '최고예영', '존나 잘만듬', '추천함', '한번 더 보고 싶다', 
        '병신 같았음', '존나 지루함', '돈 낭비', '발연기 좆노잼', '감독 머가리 분석 해야 함']

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)      # 토큰화
print('리뷰 토큰화 결과 : ', x) # [[2, 3], [4], [1, 5], [6], [7, 8, 9, 10], [11, 12], [1, 13], [14, 15], [16, 17], [18, 19, 20, 21, 22]]

# 시퀀스 데이터를 RNN 딥러닝 모델에 넣기 전에 길이를 맞추는 작업 필요
padded_x = pad_sequences(x, maxlen=5, padding='pre')
print('패딩 결과 : ', padded_x)
"""
패딩 결과 :  [[ 0  0  0  2  3]
            [ 0  0  0  0  4]
            [ 0  0  0  1  5]
            [ 0  0  0  0  6]
            [ 0  7  8  9 10]
            [ 0  0  0 11 12]
            [ 0  0  0  1 13]
            [ 0  0  0 14 15]
            [ 0  0  0 16 17]
            [18 19 20 21 22]]
"""

# 딥러닝 모델 처리
word_size = len(token.word_index) + 1   #단어 집합의 갯수

model = Sequential()
model.add(Input(shape=(5,)))
model.add(Embedding(input_dim=word_size, output_dim=12))
model.add(LSTM(32, activation='tanh'))
# model.add(Flatten)
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 5, 12)             276

 lstm (LSTM)                 (None, 32)                5760

 dense (Dense)               (None, 32)                1056

 dense_1 (Dense)             (None, 32)                1056

=================================================================
Total params: 8,148
Trainable params: 8,148
Non-trainable params: 0
_________________________________________________________________

"""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=padded_x, y=labels, epochs=30, batch_size=32, verbose=1)

loss, acc = model.evaluate(padded_x, labels, verbose=0)
print('정확도 : ', acc)

preds = model.predict(padded_x, verbose=0)
y_hat = (preds > 0.5).astype(int).ravel()
print('예측 : ', y_hat.tolist())