import numpy as np

#1. 데이터
x_train = np.load('../data/npy/cifar10_x_train.npy')
y_train = np.load('../data/npy/cifar10_y_train.npy')
x_test = np.load('../data/npy/cifar10_x_test.npy')
y_test = np.load('../data/npy/cifar10_y_test.npy')

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#1.1 데이터 전처리
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])/225.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])/225.
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3])/225.

print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape) 
# (40000, 32, 32, 3)
# (10000, 32, 32, 3)
# (10000, 32, 32, 3)

#tensorflow _to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (32, 32, 3))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(200, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Flatten())
model.add(Dense(200, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
modelpath = '../data/modelCheckpoint/k46_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('cost_loss') #loss, cost
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()

plt.title('cost_mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

# loss :  [1.2488133907318115, 0.5805000066757202]
# [[2.71783862e-02 6.36730939e-02 5.70346788e-02 2.66156107e-01
#   1.96447354e-02 1.89688474e-01 8.19806382e-02 1.17563168e-02
#   2.57382542e-01 2.55050138e-02]
#  [7.50222683e-01 3.41247469e-02 4.12743248e-04 1.29580176e-05
#   1.07158536e-04 9.82635015e-07 2.94917045e-05 3.46742945e-06
#   1.87367052e-01 2.77186260e-02]
#  [5.43333232e-01 1.97881088e-01 5.27605880e-04 9.50220492e-05
#   5.23085298e-04 1.44777714e-05 3.05963185e-05 1.04804794e-05
#   2.08373785e-01 4.92104851e-02]
#  [3.41723502e-01 3.27138510e-03 2.14675590e-02 1.23816333e-03
#   1.84665881e-02 6.40475249e-04 1.85627025e-04 6.69358473e-04
#   6.02862954e-01 9.47438087e-03]
#  [9.61820874e-03 6.40054420e-02 8.58153775e-02 3.68255228e-02
#   2.92143226e-01 2.24672575e-02 4.51423854e-01 1.45634534e-02
#   9.18460358e-03 1.39529761e-02]
#  [4.69024666e-03 5.34594292e-04 1.54371001e-02 9.48840752e-03
#   6.05379930e-03 1.31606590e-03 9.59228337e-01 5.22676019e-05
#   1.73977530e-03 1.45939423e-03]
#  [4.82562296e-02 5.54770172e-01 3.00240121e-03 1.19649414e-02
#   1.69124838e-03 1.37243820e-02 1.01534976e-02 3.25409300e-03
#   7.08708391e-02 2.82312185e-01]
#  [1.22325227e-03 2.05026936e-06 1.10069094e-02 8.61483440e-03
#   1.25374377e-03 1.21315545e-03 9.76578355e-01 3.02153103e-06
#   3.86367792e-05 6.60446240e-05]
#  [4.07202728e-02 5.37506444e-03 3.68326098e-01 1.74243182e-01
#   1.89662308e-01 1.33993804e-01 3.81956510e-02 3.91488411e-02
#   5.40865492e-03 4.92608547e-03]
#  [4.39842641e-02 5.69718421e-01 5.22959046e-04 1.00241276e-03
#   2.05026660e-03 7.71234045e-04 2.43127928e-03 4.80955321e-04
#   3.15162629e-01 6.38755411e-02]]
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [3 8 8 0 6 6 1 6 3 1]