"""
텐서플로우 기초
"""
import tensorflow as tf
print('즉시 실행 모드 : ', tf.executing_eagerly())

#Tensor 생성
print(tf.constant(1), type(tf.constant(1)))     # OD Tensor - scala
print(tf.constant([1]), type(tf.constant([1]))) # 1차원 텐서
print(tf.constant([[1]]), type(tf.constant()))