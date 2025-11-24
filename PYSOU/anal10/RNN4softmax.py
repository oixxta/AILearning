"""
RNN으로 다항 분류 : 텍스트 생성 (단어 단위)
"""
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import pad_sequences, to_categorical
from keras.layers import LSTM, Dense, Embedding, Input, Flatten
import numpy as np

#text = """경마장에 있는 말이 뛰고 있다
#그의 말이 법이다
#가는 말이 고와야 오는 말이 곱다"""

text = """지구와 같은 행성의 밝기는 지구 대기와 표면에 도달하는 햇빛이 얼마나 반사되는지에 달려 있다. 빛이 반사되는 비율을 알베도라고 하는데, 현재 지구의 평균 알베도는 0.3(30%)이다. 태양에서 오는 빛의 30%는 반사하고, 70%는 흡수한다는 뜻이다. 지구 표면의 60%를 덮고 있는 구름이 가장 큰 영향을 끼친다.
행성 전체가 두꺼운 구름으로 덮인 금성은 알베도가 0.76으로 지구보다 훨씬 밝다. 반면 보름달은 매우 밝아보이지만 알베도가 0.12(12%)로 지구에 비하면 매우 어두운 천체다.
지구에서 눈과 얼음, 구름은 주로 빛을 반사하고, 바다와 숲, 도시의 아스팔트는 주로 빛을 흡수한다. 따라서 인간 활동에 의해 지구 기온이 상승해 북극의 얼음과 빙하가 녹으면, 빛을 많이 반사하는 흰색 표면이 적어져 지구의 밝기는 어두워지고 지구 기온은 더욱 올라가는 악순환을 부를 수 있다.
실제로 지구 궤도를 도는 위성으로 측정한 데이터에 따르면 지구가 갈수록 어두워지고 있다. 2021년 ‘지구물리학 연구 서한’(Geophysical Research Letters)에 발표된 연구에 따르면 1998~2017년 사이에 지구의 알베도는 0.5% 감소했다.
과학자들은 그러나 북반구의 육지와 남반구의 바다와 얼음, 그리고 대기 중의 구름 분포가 절묘하게 균형을 이뤄 두 반구의 반사율은 거의 같다고 생각했다.
지구 표면의 색상 분포. 색상이 어두울수록 햇빛을 많이 흡수한다.
지구 기상 시스템 교란 요인 될 수도
그런데 미국 항공우주국 랭글리연구센터가 중심이 된 과학자들이 최근 미 국립과학원회보(PNAS)에 발표한 논문에 따르면 북반구가 어두워지는 정도가 남반구보다 더 심한 것으로 나타났다. 북반구가 더 많은 햇빛을 흡수하고 있다는 것이다.
이는 전 세계인의 약 90%가 거주하고 있는 북반구의 지구 기온을 더욱 높이고, 지구 전체 기상 시스템의 균형을 깨뜨리는 요인이 될 수 있다.
연구진은 2001~2024년 3개의 세레스(CERES=구름과 지구 복사 에너지 시스템) 관측 위성에서 수집한 데이터를 기반으로 지난 24년 동안 지구 밝기가 어떻게 변화했는지 조사했다. 위성이 측정한 햇빛 입사량, 복사량 데이터를 고해상도 분광 이미지, 눈과 구름 지도, 컴퓨터 기후 모델과 결합한 결과, 북반구가 남반구보다 더 어두워지고 있다는 사실이 드러났다. 북반구는 남반구에 비해 10년마다 1㎡당 약 0.34와트의 태양 에너지를 더 흡수했다.
연구진은 태양 복사 에너지의 평균 흡수량이 1㎡당 240~243와트인 점을 고려하면 큰 차이가 아니라고 볼 수도 있지만 통계적으로는 유의미한 값이라고 밝혔다."""

tok = Tokenizer()          # 현재는 단어 단위, char_level=True 이면 글자 단위
tok.fit_on_texts([text])
encoded = tok.texts_to_sequences([text])[0]
print(encoded)      # [2, 3, 1, 4, 5, 6, 1, 7, 8, 1, 9, 10, 1, 11]

print(tok.word_index)
"""
{'말이': 1, '경마장에': 2, '있는': 3, '뛰고': 4, 
'있다': 5, '그의': 6, '법이다': 7, '가는': 8, 
'고와야': 9, '오는': 10, '곱다': 11}
"""
vocab_size = len(tok.word_index) + 1
print("vocab_size : ", vocab_size)       # vocab_size :  12


# 훈련 데이터 만들기
sequences = list()
for i in text.split('\n'):
    enco = tok.texts_to_sequences([i])[0]
    #print(enco)
    for j in range(1, len(enco)):   # 레이블이 없기 때문에 바로 다음 단어를 레이블로 사용하기 위함.
        sequ = enco[:j + 1]
        sequences.append(sequ)

#print(sequences)
"""
[[2, 3], [2, 3, 1], [2, 3, 1, 4], [2, 3, 1, 4, 5], [6, 1], [6, 1, 7], 
[8, 1], [8, 1, 9], [8, 1, 9, 10], [8, 1, 9, 10, 1], [8, 1, 9, 10, 1, 11]]
"""
print('학습에 참여할 샘플의 수 : ', len(sequences))     # 학습에 참여할 샘플의 수 :  11
maxlen = max(len(i) for i in sequences)
print(maxlen)                # 가장 긴 단어 : 6

pedeingSequences = pad_sequences(sequences=sequences, maxlen=maxlen, padding='pre')
print(pedeingSequences) # 패딩을 하는 이유 : 병렬연산에 용이하기 때문.
"""
[[ 0  0  0  0  2  3]
 [ 0  0  0  2  3  1]
 [ 0  0  2  3  1  4]
 [ 0  2  3  1  4  5]
 [ 0  0  0  0  6  1]
 [ 0  0  0  6  1  7]
 [ 0  0  0  0  8  1]
 [ 0  0  0  8  1  9]
 [ 0  0  8  1  9 10]
 [ 0  8  1  9 10  1]
 [ 8  1  9 10  1 11]]
"""

# feature와 label 준비
x = pedeingSequences[:, :-1]    # feature
y = pedeingSequences[:, -1]     # label
print(x)
print(y)

# 레이블 원-핫 인코딩 처리
y = to_categorical(y, num_classes=vocab_size)
print(y[:2])

# 모델 만들기
model = Sequential()
model.add(Embedding(vocab_size, 32, mask_zero=True))
model.add(LSTM(32, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
print(model.summary())
model.fit(x, y, epochs=200, verbose=2)
print('evaluate : ', model.evaluate(x, y))

# 모델을 이용해 글자를 생성하기
def sequence_generate_text(model, t, current_word, n):
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=maxlen - 1, padding='pre')
        result = np.argmax(model.predict(encoded, verbose=0), axis=-1)

        # 예측 단어 찾기
        for word, index in t.word_index.items():
            #print(word, index)
            if index == result:     # 예측한 단어와 인덱스가 동일하면 해당 단어가 예측단어 이므로 break
                break
        
        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word
    
    sentence = init_word + sentence
    return sentence


print('생성된 텍스트 : ', sequence_generate_text(model, tok, '항공우주국', 20))
print('생성된 텍스트 : ', sequence_generate_text(model, tok, '복사랑', 10))
print('생성된 텍스트 : ', sequence_generate_text(model, tok, '북반구가', 20))
print('생성된 텍스트 : ', sequence_generate_text(model, tok, '행성', 20))