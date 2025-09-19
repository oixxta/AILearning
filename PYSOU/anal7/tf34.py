"""
MNIST 데이터 세트의 데이터들을 전처리 과정에서 섞어보기(Slice), GradientTape를 한 CNN 모델

Model Subclassing API 사용
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 데이터 가져오기 
(x_Train, y_Train), (x_Test, y_Test) = tf.keras.datasets.mnist.load_data()


# 구조 변경(차원)
# CNN은 witdth, height, 행, channel 정보가 담겨야 하기 때문에 4개의 입력값이 필요함.
print(x_Train.shape)    # (60000, 28, 28)
x_Train = x_Train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_Test = x_Test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
print(x_Train.shape)    # (60000, 28, 28, 1)
#print(x_Train)


# 데이터 섞어보기 (Slice) : 편향된 데이터일 수도 있기 때문에 섞는 과정이 필요할 수도 있음.
#from_tensor_slices 사용.
"""
#랜덤데이터 x를 만들어서 먼저 연습해보기
x = np.random.sample((5, 2))    
print(x)
dset = tf.data.Dataset.from_tensor_slices(x)
print(dset)
dset = tf.data.Dataset.from_tensor_slices(x).shuffle(buffer_size=10000).batch(5) #buffer_size : 입력 데이터보다 커야 함.
print(dset)
for a in dset : 
    print(a)
"""
#Train Data 섞기
train_ds = tf.data.Dataset.from_tensor_slices((x_Train, y_Train)).shuffle(buffer_size=60000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_Test, y_Test)).batch(32)
print(train_ds)
print(test_ds)


# 모델 설계하기
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))
        self.flat = tf.keras.layers.Flatten(dtype='float32')
        self.d1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(rate=0.3)
        self.outputs = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.flat(net)
        net = self.drop1(net)
        net = self.outputs(net)
        return net

# 모델 학습하기 : 일반적인 방법
model = MyModel()
temp_inputs = tf.keras.Input(shape=(28, 28, 1))
model(temp_inputs)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer_object = tf.keras.optimizers.Adam()
"""
model.compile(optimizer= optimizer_object, loss = loss_object, metrics=['acc'])
model.fit(x_Train, y_Train, epochs=5, batch_size=128, verbose=2, \
          max_queue_size=10, workers=4, use_multiprocessing=True) # process 기반 스레딩 처리

# 모델 평가하기
score = model.evaluate(x_Test, y_Test)
print('test loss : ', score[0])
print('test accuracy : ', score[1])
print('예측값 : ', np.argmax(model.predict(x_Test[:2]), 1)) #예측값 :  [7 2]
print('실제값 : ', y_Test[:2])                              #실제값 :  [7 2]
"""

# 모델 학습하기 : Gradiant tape 사용하기 - 모델 서브 프로세싱 학습 방법
#모델 손실과 성능을 측정할 지표 선택. 수집된 측정 지표를 바탕으로 최종 결과 출력을 위한 객체 생성.
train_loss = tf.keras.metrics.Mean()    # 주어진 값의 (가중)평균을 계산
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()    # 주어진 값의 (가중)평균을 계산
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(images, labels): # 얘를 반복하면 loss를 최소화 하게 됨.
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)  # loss 최소화를 위한 미분 계산.
    optimizer_object.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
    

def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    for train_images, train_labels in train_ds:
        train_step(train_images, train_labels)

    for test_images, test_labels in test_ds:
        train_step(test_images, test_labels)

    template = 'epochs : {}, train_loss : {}, train_acc : {}, test_loss : {}, test_acc : {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))