"""
자연어 처리

Token, Vector, Embedding 이해하기
단어를 수치화 하기.

Onehot Encoding
"""

print('레이블 인코딩 (수동)')
datas = ['python', 'len', 'program', 'computer', 'say']
sorted_datas = sorted(datas)
print('정렬된 데이터 : ', sorted_datas)     # 정렬된 데이터 :  ['computer', 'len', 'program', 'python', 'say']
manual_labels = list(range(len(sorted_datas)))
print('인덱스 라벨 : ', manual_labels)      # 인덱스 라벨 :  [0, 1, 2, 3, 4]

### 레이블 인코더 사용
import numpy as np
onehot_manual = np.eye(len(manual_labels))
print(onehot_manual)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
"""

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder_labels = encoder.fit_transform(datas)
print(encoder.classes_)             # ['computer' 'len' 'program' 'python' 'say']
print(datas)                        # ['python', 'len', 'program', 'computer', 'say']
print(encoder_labels)               # [3 1 2 0 4]


### 원-핫 인코더 사용
from sklearn.preprocessing import OneHotEncoder     # 원핫 인코더를 써서 사용.
sorted_datas_2d = np.array(sorted_datas).reshape(-1, 1)        # 2차원화 시키기
onehot_encoder = OneHotEncoder(sparse_output=False)            # nd array 형태로 반환시키기
onehot_encoded = onehot_encoder.fit_transform(sorted_datas_2d)
print(onehot_encoded)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
"""


### 판다스 사용
import pandas as pd
df = pd.DataFrame({'datas' : sorted_datas})
onehot_df = pd.get_dummies(df, dtype=int)
print(onehot_df)
"""
   datas_computer  datas_len  datas_program  datas_python  datas_say
0               1          0              0             0          0
1               0          1              0             0          0
2               0          0              1             0          0
3               0          0              0             1          0
4               0          0              0             0          1
"""