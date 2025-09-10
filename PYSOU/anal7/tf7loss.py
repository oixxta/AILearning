"""
cost를 최소화 하기!
"""
import tensorflow as tf
import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5]     #feature
y = [1, 2, 3, 4, 5]     #label
bias = 0                #절편

w_val = []
cost_val = []
for i in range(-30, 50):
    feed_w = i * 0.1
    hypothesis = tf.multiply(feed_w, x) + bias        # y = wx + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))       # cost 수식 적용
    w_val.append(feed_w)
    cost_val.append(cost)
    #print(str(i) + ' ' + 'cost : ' + str(cost.numpy()) + ' ' + 'weight : ' + str(feed_w))

plt.plot(w_val, cost_val)
plt.xlabel('weight')
plt.ylabel('cost')
plt.show()
plt.close()