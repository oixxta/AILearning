"""
텐서플로우 연산자
"""
import tensorflow as tf
import numpy as np

print(tf.__version__)        #설치 정상, 버전 2.20.0
print('GPU', '사용가능' if tf.config.list_physical_devices('GPU') else '불가능')
print('즉시 실행 모드 : ', tf.executing_eagerly())

def func1():
    #텐서플로우 연산자
    x = tf.constant(7)
    y = tf.constant(3)

    #삼항연산 : cond (컨디션)
    result1 = tf.cond(x > y, lambda:tf.add(x, y), lambda:tf.subtract(x, y))
    print(result1)

    #case 조건
    f1 = lambda:tf.constant(1)
    print(f1)
    f2 = lambda:tf.constant(2)
    a = tf.constant(3)
    b = tf.constant(4)
    result2 = tf.case([(tf.less(a, b), f1)], default=f2)      #less : 두 개중 작은쪽 반환
    print(result2)

    #관계연산
    print(tf.equal(1, 2))
    print(tf.not_equal(1, 2))
    print(tf.greater(1, 2))
    print(tf.greater_equal(1, 2))
    print(tf.less(1, 2))

    #논리연산
    print(tf.logical_and(True, False))
    print(tf.logical_or(True, False))
    print(tf.logical_not(True))

    #유일 합집합 (유니크)
    kbs = tf.constant([1, 2, 2, 2, 3])
    val, index = tf.unique(kbs)     #유일값과 유일값에 대한 인덱스 반환
    print(val.numpy())
    print(index.numpy())

    #reduce : 차원 축소
    ar = [[1,2], [3,4]]
    print(tf.reduce_mean(ar).numpy())
    print(tf.reduce_mean(ar, axis=0).numpy())   #열 기준
    print(tf.reduce_mean(ar, axis=1).numpy())   #행 기준



#func1()