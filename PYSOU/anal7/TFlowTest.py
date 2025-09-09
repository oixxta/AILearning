"""
텐서플로우 테스트 : 제대로 설치되었는지 확인.

터미널 창에 pip install tensorflow
"""
import tensorflow as tf
print(tf.__version__)        #설치 정상, 버전 2.20.0

print('GPU', '사용가능' if tf.config.list_physical_devices('GPU') else '불가능')

print('즉시 실행 모드 : ', tf.executing_eagerly())

#Tensor 생성
print(tf.constant(1), type(tf.constant(1)))       # OD Tsnsor - scala
print(tf.constant([1]), type(tf.constant([1])))   # 1차원 텐서
print(tf.constant([[1]]), type(tf.constant([[1]]))) # 2차원 텐서
print(tf.constant([[[1]]]), type(tf.constant([[[1]]]))) #3차원 텐서

print()
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c)                  # tf.Tensor([4 6], shape=(2,), dtype=int32)
d = tf.constant([3])      #브로드케스팅이 발생함.
e = c + d                 # 파이썬식 더하기, 속도 느림.
print(e)                  # tf.Tensor([7 9], shape=(2,), dtype=int32)
f = tf.add(c, d)          # 텐서플로우식 더하기, 속도가 더 빠름.
print(f)                  # tf.Tensor([7 9], shape=(2,), dtype=int32)


print(7)
# 파이썬의 상수 7을 텐서로 형변환 : 4가지 방법, 같은 결과
print(tf.convert_to_tensor(7, dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7, dtype=tf.float32))
#파이썬의 데이터와 텐서의 데이터는 다름. 형변환 필요함.

#넘파이의 nd.array와 텐서 사이의 type 자동변환됨.
import numpy as np
arr = np.array([1, 2])
print(arr, type(arr))       # [1 2] <class 'numpy.ndarray'>