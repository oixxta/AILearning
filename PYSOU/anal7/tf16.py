"""


"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 1) 데이터 읽기
url = 'https://github.com/pykwon/python/blob/master/testdata_utf8/hd_carprice.xlsx?raw=true'
trainDf = pd.read_excel(url, sheet_name='train')
testDf  = pd.read_excel(url, sheet_name='test')

X_train = trainDf.drop(columns=['가격'])
y_train = trainDf['가격'].values  # (71,) 형태
X_test  = testDf.drop(columns=['가격'])
y_test  = testDf['가격'].values

# 2) 컬럼 타입 점검 (참고 출력)
# print(X_train.dtypes)

# 3) 전처리기: 범주형 원-핫 + 수치형 표준화
cat_cols = ['종류', '연료', '변속기']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
        ('num', StandardScaler(), num_cols),
    ],
    remainder='drop'
)

# 4) 전처리 먼저 학습/변환
preprocess.fit(X_train)
X_train_t = preprocess.transform(X_train)  # dense numpy array
X_test_t  = preprocess.transform(X_test)

# 5) 간단한 신경망 (출력층 linear) + 정규화 + 조기종료
inputs = tf.keras.layers.Input(shape=(X_train_t.shape[1],))
x = tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
x = tf.keras.layers.Dense(16, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
outputs = tf.keras.layers.Dense(1)(x)  # linear

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

model.fit(X_train_t, y_train, epochs=200, batch_size=16,
          validation_data=(X_test_t, y_test), callbacks=[es], verbose=0)

test_mse = model.evaluate(X_test_t, y_test, verbose=0)[0]
print('Test MSE:', test_mse)

# 6) 예측
y_pred = model.predict(X_test_t, verbose=0).ravel()
print('예측값(5):', y_pred[:5])
print('실제값(5):', y_test[:5])

# 7) 새 샘플 예측 — ★ 컬럼명 반드시 정확히!
newX = pd.DataFrame(
    [[2015, '준중형', 12.3, 204, 27.0, '가솔린', 0, 1591, 1300, '자동']],
    columns=['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기']
)
newX_t = preprocess.transform(newX)
new_pred = model.predict(newX_t, verbose=0).ravel()
print('새 샘플 예측:', new_pred[0])