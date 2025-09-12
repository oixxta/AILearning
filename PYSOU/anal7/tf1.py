"""
텐서플로우 기초
"""
import tensorflow as tf
import numpy as np
print('즉시 실행 모드 : ', tf.executing_eagerly())

#Tensor 생성
print(tf.constant(1), type(tf.constant(1)))     # OD Tensor - scala
print(tf.constant([1]), type(tf.constant([1]))) # 1차원 텐서
print(tf.constant([[1]]), type(tf.constant([[1]]))) # 2차원 텐서
print(tf.constant([[[1]]]), type(tf.constant([[[1]]])))  #3차원 텐서

#텐서의 연산은 파이썬 연산자도 사용 가능, 각 항의 요소를 더함.
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c)    # tf.Tensor([4 6], shape=(2,), dtype=int32)

#브로드케스팅이 발생함.(d가 [3,3]으로 간주됨.) 파이썬 연산자를 사용하는 것은 속도가 느림.
d = tf.constant([3])
e = c + d
print(e)    # tf.Tensor([7 9], shape=(2,), dtype=int32)

#텐서플로우식 더하기. 속도가 파이썬 연산자를 쓰는 것보다 빠름. 여기서도 브로드케스팅이 발생함.
f = tf.add(c, d)
print(f)    # tf.Tensor([7 9], shape=(2,), dtype=int32)

#파이썬의 상수 7을 텐서로 형변환 : 4가지 방법이 있음. 같은 결과임.
#파이썬의 데이터와 텐서의 데이터는 다름. 형변환이 필요함.
print(tf.convert_to_tensor(7, dtype=tf.float32))    #tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.cast(7, dtype=tf.float32))                 #tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.constant(7.0))                             #tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.constant(7, dtype=tf.float32))             #tf.Tensor(7.0, shape=(), dtype=float32)

#넘파이의 nd.array와 텐서 사이의 type은 자동변환됨.
arr = np.array([1, 2])
print(arr, type(arr))   #[1 2] <class 'numpy.ndarray'>
tfarr = tf.add(arr, 5)
print(tfarr)            #tf.Tensor([6 7], shape=(2,), dtype=int32), 파이썬 상수 5가 자동으로 텐서로 바뀜.
tf.print(tfarr)         #[6 7], 텐서플로우식 출력 함수 : 훨신 깔끔함.
print(tfarr.numpy())    #[6 7], 탠서타입이 일반 파이썬 타입으로 바뀜.
print(np.add(tfarr, 3)) #[ 9 10], 텐서플로우가 넘파이 타입으로 바뀜.
