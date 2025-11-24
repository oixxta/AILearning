"""
RNN (Recurrent Neural Network, 순환 참조 네트워크) 기초 모델 작성해보기

순환 신경망(RNN) 구조 이해 (네트워크 구성)
"""
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Input

model = Sequential()
model.add(Input(shape=(2, 10)))     # 입력 형태를 정의함.(시퀀스 길이, 각 시점에서의 입력 벡터 크기(feature))
# 의미 : (batchsize, 시퀀스 길이, 입력벡터 크기)
# x[
#    [0.1, 0.2, 0.3, ..., 0.10],  # 시점1
#    [0.1, 0.2, 0.3, ..., 0.10],  # 시점2
# ]
# ht = tanh(WxXt + WhHt - 1 + b)
# params = (input_dim + units) * units + units     # (10 + 3) * 3 + 3 = 39 + 3 = 42

### 심플 RNN 사용
#model.add(SimpleRNN(3))        
#print(model.summary())

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 3)                 42        

=================================================================
Total params: 42
Trainable params: 42
Non-trainable params: 0
_________________________________________________________________

입력값에 대해서 현재 상태가 다음 상태에 영항을 줌.
노드가 자기 자신을 참조함.
"""

### LSTM 사용
#model.add(LSTM(units=3))        # 4(게이트 수) * (10 + 3 + 1) * 3 = 168
#print(model.summary())

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 3)                 168

=================================================================
Total params: 168
Trainable params: 168
Non-trainable params: 0
_________________________________________________________________

SimpleRNN과 비교해서 Param의 갯수가 대폭 증가함.
내부적으로 units=3이지만, 게이트의 수가 1개 더 증가했음.
"""

### LSTM 사용 + 배치사이즈 정의
model2 = Sequential()
model2.add(Input(batch_size=8, shape=(2, 10)))      # 입력텐서(8, 2, 10)
model2.add(SimpleRNN(units=3, return_sequences=True))   #
print(model2.summary())

"""
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (8, 2, 3)                 42

=================================================================
Total params: 42
Trainable params: 42
Non-trainable params: 0
_________________________________________________________________
"""