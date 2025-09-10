"""
tf8에서 이어짐.
텐서플로우에서는 자동미분에 의한 GradiantTape을 제공함.
"""
import tensorflow as tf
import numpy as np

x = tf.Variable(5.0)
w = tf.Variable(0.0)

@tf.function
def train_step():
    # Gradianttape : 연산 과정을 기억해 두었다가, 나중에 자동으로 Gradiant를 계산하기 위한 미분을 실시.
    with tf.GradientTape() as tape:
        y = tf.multiply(w, x)
        loss = tf.square(tf.subtract(y, 50))
    grad = tape.gradient(loss, w)       #자동으로 미분 계산됨. loss를 w로 미분
    mu = 0.01   # 학습률
    w.assign_sub(mu * grad)
    return loss         #손실값 반환

for i in range(10):
    loss = train_step()
    print('{:1}, w: {:5f}, loss : {:.5f}'.format(i, w.numpy(), loss.numpy()))

"""
0, w: 5.000000, loss : 2500.00000
1, w: 7.500000, loss : 625.00000
2, w: 8.750000, loss : 156.25000
3, w: 9.375000, loss : 39.06250
4, w: 9.687500, loss : 9.76562
5, w: 9.843750, loss : 2.44141
6, w: 9.921875, loss : 0.61035
7, w: 9.960938, loss : 0.15259
8, w: 9.980469, loss : 0.03815
9, w: 9.990234, loss : 0.00954
"""

# keras.optimizers 패키지에 있는 Adam, SGD, RMSprop... 사용해보기
opti = tf.keras.optimizers.SGD(learning_rate=0.01)

x = tf.Variable(5.0)
w = tf.Variable(0.0)

@tf.function
def train_step2():
    # Gradianttape : 연산 과정을 기억해 두었다가, 나중에 자동으로 Gradiant를 계산하기 위한 미분을 실시.
    with tf.GradientTape() as tape:
        y = tf.multiply(w, x)
        loss = tf.square(tf.subtract(y, 50))
    grad = tape.gradient(loss, w)       #자동으로 미분 계산됨. loss를 w로 미분
    opti.apply_gradients([(grad, w)])   #w와 D의 값이 자동으로 갱신.
    return loss         #손실값 반환

for i in range(10):
    loss = train_step2()
    print('{:1}, w: {:5f}, loss : {:.5f}'.format(i, w.numpy(), loss.numpy()))
"""
0, w: 5.000000, loss : 2500.00000
1, w: 7.500000, loss : 625.00000
2, w: 8.750000, loss : 156.25000
3, w: 9.375000, loss : 39.06250
4, w: 9.687500, loss : 9.76562
5, w: 9.843750, loss : 2.44141
6, w: 9.921875, loss : 0.61035
7, w: 9.960938, loss : 0.15259
8, w: 9.980469, loss : 0.03815
9, w: 9.990234, loss : 0.00954
"""

# 선형회귀 모형을 만들어 보기
# keras.optimizers 패키지에 있는 Adam, SGD, RMSprop... 사용해보기
opti = tf.keras.optimizers.SGD(learning_rate=0.01)

tf.random.set_seed(2)
w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))

@tf.function
def train_step3(x, y):
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
    grad = tape.gradient(loss, [w, b])       #자동으로 미분 계산됨. loss를 w와 b로 미분

    opti.apply_gradients(zip(grad, [w, b]))   #w와 D의 값이 자동으로 갱신.
    return loss         #손실값 반환

x = [1., 2., 3., 4., 5.]    #feature
y = [1.2, 2.0, 3.0, 3.5, 5.5]   #label

w_vals = []
cost_vals = []

for i in range(1, 101):
    cost_val = train_step3(x, y)
    cost_vals.append(cost_val.numpy())
    w_vals.append(w.numpy())
    if i % 10 == 0:
        print(cost_val)

print(cost_vals)
print(w_vals)

import matplotlib.pylab as plt
plt.plot(w_vals, cost_vals, 'o--')
plt.xlabel('w')
plt.ylabel('cost')
plt.legend()
plt.show()
plt.close()

print('cost가 최소값일때 w의 값 : ', w.numpy())
print('cost가 최소값일때 b의 값 : ', b.numpy())

y_pred = tf.multiply(x, w) + b
# 기존값으로 예측해보기
print('y_pred : ', y_pred)
plt.plot(x, y, 'ro', label='real')
plt.plot(x, y_pred, 'b-', label='pred')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.close()

# 새값으로 예측해보기
new_x = [3.5, 9.0]
new_pred = tf.multiply(new_x, w) + b
print('예측결과 : ', new_pred.numpy())  #예측결과 :  [3.5472796 8.988613 ]
