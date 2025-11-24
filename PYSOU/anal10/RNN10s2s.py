import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. 병렬 문장 데이터
# data는 한국어 질문(또는 인사말)과 영어 답변의 쌍으로 이루어진 리스트.
# Seq2Seq 모델이 “입력 시퀀스 → 출력 시퀀스” 형태로 학습할 수 있게 해 줌.
data = [
    ("안녕",      "hi"),
    ("잘 지내?",  "how are you?"),
    ("고마워",    "thank you"),
    ("좋은 아침", "good morning"),
    ("사랑해",    "i love you"),
    ("잘 자",     "good night"),
]

# 2. 토크나이저 함수
# 가장 빈도 높은 100개의 단어만 사전에 남기겠다는 뜻(나머지는 OOV 처리).
def tokenize(sentences, num_words=100):
    tok = Tokenizer(num_words=num_words, filters='') # Tokenizer: 텍스트를 정수 시퀀스로 변환
    # filters=''로 설정하면 기본 필터(!"#$%&()*+,…)를 사용하지 않고, 공백 기준으로 토큰 분리.

    tok.fit_on_texts(sentences)  # 단어 사전(단어 → 정수 인덱스)을 만듦
    seqs = tok.texts_to_sequences(sentences)  # 각 문장을 정수 인덱스의 리스트(시퀀스)로 변환
    return seqs, tok
      # seqs: 문장별로 정수 인덱스 시퀀스를 담은 리스트
      # tok: 구축된 토크나이저 객체 (인덱스→단어 매핑 등 속성 포함)

# 3. 입력/출력 문장 분리
# input_texts: Seq2Seq 인코더에 들어갈 한국어 시퀀스만 뽑아 리스트로 저장
# target_texts: Seq2Seq 디코더에 들어갈 영어 시퀀스에
# <sos> (start-of-sequence) 토큰을 앞에 붙이고, <eos> (end-of-sequence) 토큰을 뒤에 붙인다.
# 디코더 훈련 시, <sos>에서 시작해 단어를 하나씩 예측후, <eos>가 나오면 시퀀스 끝을 알 수 있도록 함.
input_texts  = [kor for kor, eng in data]
target_texts = ['<sos> ' + eng + ' <eos>' for kor, eng in data]

# 4. 토크나이징
# 인코더용 한국어 문장(input_texts)을 tokenize()에 넘겨서 encoder_seqs에는 
# 한국어 각 문장의 정수 인덱스 시퀀스 리스트가 enc_tok에는 한국어용 토크나이저가 저장
encoder_seqs, enc_tok = tokenize(input_texts)

# 디코더용 영어 문장(target_texts)을 tokenize()에 넘겨서 decoder_seqs에는 
# 영어 각 문장의 정수 시퀀스(시작/종료 토큰 포함) dec_tok에는 영어용 토크나이저가 저장
# 이렇게 얻은 시퀀스들은 이후 pad_sequences를 거쳐 LSTM이 처리할 수 있는 동일 길이의 배열이 됨.
decoder_seqs, dec_tok = tokenize(target_texts)
# 각 토크나이저 객체(enc_tok, dec_tok)는 정수 ↔ 단어 매핑을 담당하게 된다

# 5. 패딩 : 길이가 다른 문장들을 동일한 길이로 맞추기 위해 뒤에 0으로 패딩
encoder_input_data = pad_sequences(encoder_seqs, padding='post')
decoder_sequences   = pad_sequences(decoder_seqs, padding='post')

# 6. decoder 입력/타겟 분리
# 마지막 토큰 <eos> (또는 패딩된 0)를 제거한 입력 시퀀스.
# 모델은 <sos> hi 를 입력으로 받아 다음 단어 hi, eos를 예측하게 된다.
decoder_input_data  = decoder_sequences[:, :-1] 

# 첫 번째 <sos>를 제외한 정답 시퀀스다. 즉, 모델이 예측해야 할 출력 시퀀스.
decoder_target_data = decoder_sequences[:,  1:]

# sparse_categorical_crossentropy를 사용할 경우 출력 shape은 (batch, timesteps, 1) 이어야 함.
# [..., np.newaxis]는 마지막 축에 차원을 추가한다.
decoder_target_data = decoder_target_data[..., np.newaxis]  # (batch, timesteps, 1)
    # decoder_sequences	<sos> i love you <eos> → [1, 5, 6, 7, 2]	전체
    # decoder_input_data	<sos> i love you → [1, 5, 6, 7]	입력
    # decoder_target_data	i love you <eos> → [5, 6, 7, 2]	정답

# 7. Seq2Seq 모델의 임베딩 및 LSTM 구조를 설정하는 핵심 하이퍼파라미터 정의
# enc_tok.word_index: 입력 문장(한국어)의 단어 → 인덱스 사전을 의미.
# 예: { "안녕": 1, "잘": 2, "지내": 3, ... }, len(...): 총 등장한 고유 단어 수
# +1: 패딩(PAD)을 위한 0번 인덱스를 고려해 단어 수를 하나 더 추가
enc_vocab = len(enc_tok.word_index) + 1   # 인코더 Embedding 레이어의 input_dim으로 사용

# 출력 문장(영어) 쪽의 단어 사전 크기를 의미. 역시 +1은 0번 인덱스(pad)를 위한 자리
dec_vocab   = len(dec_tok.word_index) + 1   # 디코더 Embedding 및 마지막 Dense의 출력크기로 사용
hidden_size = 64  # LSTM의 은닉 상태 벡터의 차원 수. 
                  # 작을수록 빠르고 간단하며, 클수록 표현력이 좋아지지만 계산량도 많아진다.

# 8. 학습용 Seq2Seq 모델 정의
# 8.1 Encoder 목적 : 입력 시퀀스(예: 한국어 문장)를 받아서 요약된 의미 벡터인 
# state_h (hidden state)와 state_c (cell state)를 얻는 것이다.
# 이 상태 정보는 디코더가 답변을 생성하는 데 필요한 문맥이 된다.
encoder_inputs = Input(shape=(None,), name='encoder_inputs')  # (None,) 입력 길이 가변적 의미

# 단어 인덱스를 의미 벡터로 변환하는 임베딩 레이어. 예: [1, 5, 9] → [[0.1, -0.3, ...],...
enc_emb_layer = tf.keras.layers.Embedding(enc_vocab, hidden_size, name='enc_emb')

# 입력된 단어 인덱스 시퀀스를 임베딩 벡터 시퀀스로 변환
# 결과 shape: (batch_size, timesteps, hidden_size)  예: [1, 2, 3] → 3개의 (64차원) 벡터
encoder_emb = enc_emb_layer(encoder_inputs)

# return_state=True: 최종 시점의 hidden state (state_h)와 cell state (state_c)를 반환
encoder_lstm = LSTM(hidden_size, return_state=True, name='encoder_lstm')

# encoder_emb을 LSTM에 입력
# _: 전체 출력 시퀀스 (디코더에서 안 쓰므로 무시)
# state_h: 마지막 시점의 hidden state (디코더의 초기 hidden state로 사용)
# state_c: 마지막 시점의 cell state (디코더의 초기 cell state로 사용)
_, state_h, state_c = encoder_lstm(encoder_emb)
encoder_states = [state_h, state_c]   # 디코더에 넘겨줄 상태 정보를 리스트로 묶음
# [단어 인덱스 시퀀스] -> [Embedding Layer] -> [LSTM Layer] ->[state_h, state_c] -> 디코더로 전달
# "안녕" ->	[1]	[0.25, -0.31, ...] ->	state_h, state_c -> 디코더로 전달됨

# 8.2 Decoder : 인코더에서 전달받은 상태로 <sos>부터 시작해 하나씩 단어를 생성해가는 구조
# 디코더의 입력은 영어 정답 문장 앞에 <sos>를 붙인 시퀀스. 예: "i love you" → <sos> i love you
# shape=(None,)은 시퀀스 길이가 가변적임을 나타낸다.
decoder_inputs = Input(shape=(None,), name='decoder_inputs')

# 디코더의 입력 인덱스를 의미 벡터로 바꿔주는 임베딩 레이어. dec_vocab: 디코더 단어 사전 크기
# hidden_size: 각 단어를 몇 차원으로 표현할지 (LSTM hidden size와 동일하게 설정)
dec_emb_layer  = tf.keras.layers.Embedding(dec_vocab, hidden_size, name='dec_emb')

# 실제 입력 시퀀스를 임베딩하여 벡터 시퀀스로 변환. 예:[1, 5, 9] → [[0.1, -0.3, ...],..., [...]]
decoder_emb = dec_emb_layer(decoder_inputs)

# LSTM을 한 타임스텝씩 돌면서 출력을 생성하는 구조
  # return_sequences=True: 모든 시점의 출력 벡터를 반환 (→ 전체 문장 예측을 위해 필요)
  # return_state=True: 마지막 시점의 상태도 반환 (인퍼런스 때 사용)
decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')

# 디코더의 LSTM에 decoder_emb를 입력하고, 초기 상태는 인코더에서 나온 encoder_states ([state_h, state_c])로 설정
  # 출력은: decoder_outputs: 모든 시점의 출력 시퀀스 (예: "i", "love", "you")
  # _, _: 마지막 시점의 상태 (여기서는 학습 시점이므로 사용하지 않음)
decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)

# 각 시점의 LSTM 출력(hidden state)을 단어 확률 분포로 변환
# 출력 벡터 크기는 dec_vocab (모든 단어에 대한 softmax) 각 시점에서 가장 높은 확률을 가진 단어 선택
decoder_dense  = Dense(dec_vocab, activation='softmax', name='decoder_dense')

# 위에서 나온 decoder_outputs를 Dense layer에 통과시켜 각 시점마다 단어 예측 결과를 얻음
decoder_outputs = decoder_dense(decoder_outputs)

# 8.3 모델 컴파일/학습 : 델을 학습시키는 최종 단계
# 입력: encoder_inputs: 인코더에 들어갈 입력 시퀀스 (한국어 등 원문)
#       decoder_inputs: 디코더에 들어갈 입력 시퀀스 (<sos> + 정답 문장)
# 출력: decoder_outputs: 디코더가 예측한 단어 분포 (softmax 결과)
train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# 'sparse_categorical_crossentropy' : 다중 클래스 분류용 손실 함수
# 출력이 softmax이고, 정답이 원-핫 벡터가 아니라 정수 인덱스일 때 사용.
train_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
train_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,         # 디코더가 예측해야 할 정답 (예:i love you <eos> → [5,6,7,2])
    batch_size=2, epochs=300, verbose=2   # 한 번에 두 개 문장씩 학습
)
print("학습 완료")

# 9. 인퍼런스용 모델 분리
# 학습된 Seq2Seq 모델을 "추론(Inference)"용으로 분리하는 작업의 첫 단계
# 학습 시에는 "정답 문장 전체"를 알고 있으므로, 디코더에 한꺼번에 넣고 학습한다.
# 추론 때는 <sos>로 시작해서 단어를 하나씩 예측하며 자기가 생성한 단어를 다시 디코더에 넣어야 함.
# 이 때문에 학습 모델과 다른 방식으로 작동하는 "인퍼런스 전용 모델"을 따로 구성해야 한다.
# 9.1 Encoder 모델 (입력 문장 → 문맥 상태 추출)
# 입력: encoder_inputs (예: "안녕" → [1])
# 출력: encoder_states = [state_h, state_c]
# 목적: 입력 문장을 인코딩하여 디코더에 넘겨줄 context (문맥 정보)를 생성
encoder_model = Model(encoder_inputs, encoder_states)

# 9.2 Decoder 모델
# -- 상태 입력 레이어 (상태 입력 정의)
  # 추론 시에는 매 스텝마다 디코더에 상태(state)를 직접 전달해야 하므로,
  # 이전 타임스텝의 hidden state와 cell state를 입력으로 받을 수 있게 정의함.
decoder_state_input_h = Input(shape=(hidden_size,), name='dec_state_h')  # 이전 hidden state
decoder_state_input_c = Input(shape=(hidden_size,), name='dec_state_c')   # 이전 cell state
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# -- 임베딩 & LSTM (인퍼런스용) : 디코더의 인퍼런스(추론) 전용 모델을 완성하는 부분
# 학습 시 사용한 임베딩 레이어(dec_emb_layer)를 재사용한다.
# 디코더의 입력 단어(예: <sos>, thank 등)를 벡터로 변환한다.
dec_emb2 = dec_emb_layer(decoder_inputs)   # decoder_inputs는 (1, 1) 형태의 단어 인덱스 하나만 들어옴. step-by-step

# LSTM 실행 (상태를 외부에서 입력)
# 이전 상태 decoder_states_inputs = [state_h, state_c]를 넣고, 현재 입력(decoder_inputs)에 대한 출력생성.
# 출력은: decoder_outputs2: 현재 스텝의 단어 확률, state_h2, state_c2: 다음 스텝에 넘길 새로운 상태
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)

# LSTM의 출력을 Dense(dec_vocab, softmax)를 통해 전체 단어 사전에 대한 확률 분포로 변환한다.
# 예: [0.01, 0.03, 0.88, 0.02, ...] → 가장 높은 확률을 가진 인덱스를 추출해 번역 단어 생성
decoder_outputs2 = decoder_dense(decoder_outputs2)   # 단어 확률 → 단어 인덱스
decoder_model = Model(   # 디코더 모델 정의
       # 입력: 현재 단어 인덱스 (decoder_inputs), 이전 상태 (state_h, state_c)
    [decoder_inputs] + decoder_states_inputs,
       # 출력: 현재 단어의 softmax 확률분포 (decoder_outputs2), 다음 상태 (state_h2, state_c2)                       
    [decoder_outputs2, state_h2, state_c2]
)

# 10. 번역 함수 : Seq2Seq 모델 + 디코더 인퍼런스 모델로, 한글 입력 문장을 영어로 번역
def translate(sentence):
    # 10.1 인코더로부터 상태 획득
    # 인코더에 입력하면, 문맥 상태 벡터 state_h, state_c 반환. 예:"고마워" → state_h, state_c
    seq = enc_tok.texts_to_sequences( [ sentence ] )  # enc_tok으로 한글 문장을 정수 시퀀스로 변환

    seq = pad_sequences(seq, maxlen=encoder_input_data.shape[1], padding='post')
    states = encoder_model.predict(seq)

    # 10.2 디코더 초기 입력: <sos>
    # 번역시작을 알리는 <sos>토큰을 입력으로 넣고, 빈리스트 decoded에 예측된 단어들 누적저장 준비
    target_seq = np.array( [ [ dec_tok.word_index['<sos>'] ] ] )
    decoded = [ ]

    # 10.3 토큰 생성 루프 : (한 단어씩 생성)
    while True:
        # 현재 입력 토큰과 상태를 디코더에 넣어, 다음 단어의 확률 분포와 다음 상태를 얻음
        output_tokens, h, c = decoder_model.predict([target_seq] + states)
        # 가장 확률 높은 단어 인덱스를 선택 → 단어로 변환
        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = dec_tok.index_word.get(sampled_idx, '')

        # <eos>(종료 토큰)이 나오거나 10개 초과하면 종료
        if sampled_word == '<eos>' or len(decoded) > 10:
            break
        decoded.append(sampled_word)

        # 다음 step 준비 및 상태 업데이트
        target_seq = np.array( [ [ sampled_idx ] ] )   # 선택된 단어를 다음 입력으로 사용
        states = [h, c]      # 상태도 업데이트해서 이어지는 문맥 유지

    return ' '.join(decoded)  # 최종 번역결과 반환. 예:['thank', 'you'] → "thank you"

# 11. 테스트
print("\n번역 테스트:")
for s in input_texts:
    print(f"{s} ➜ {translate(s)}")