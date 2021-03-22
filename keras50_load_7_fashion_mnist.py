#fashion_mnist CNN구현


import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.load('../data/npy/fashion_mnist_x_train.npy')
y_train = np.load('../data/npy/fashion_mnist_y_train.npy')
x_test = np.load('../data/npy/fashion_mnist_x_test.npy')
y_test = np.load('../data/npy/fashion_mnist_y_test.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#이미지 보기
#plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show(xs


#1.1 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)/225
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)/225
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

print(x_train.shape) # (48000, 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)
print(x_val.shape) # (12000, 28, 28, 1)

#tensorflow _to_categorica
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
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
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k46_fashion_{epoch:02d}-{val_loss:4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

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

# loss :  [0.40063905715942383, 0.8716999888420105]
# [[2.48733933e-12 1.07870179e-09 1.15780502e-14 6.78643863e-10
#   1.51395951e-09 1.12935365e-03 1.68877565e-10 7.36090913e-03
#   1.86516857e-07 9.91509557e-01]
#  [2.13630307e-15 6.47234191e-19 9.99917746e-01 1.63857285e-15
#   8.22655711e-05 1.48349221e-31 3.80181042e-08 2.12544440e-29
#   9.03915762e-16 1.48085746e-31]
#  [1.59954733e-19 1.00000000e+00 1.42639586e-25 1.06429793e-18
#   2.53062239e-18 0.00000000e+00 4.22122881e-19 1.01963897e-35
#   1.21725431e-38 0.00000000e+00]
#  [2.06046230e-16 1.00000000e+00 1.66212907e-21 1.46059143e-15
#   1.23334752e-15 0.00000000e+00 4.94531928e-16 1.35649671e-29
#   4.52083402e-32 2.47108341e-35]
#  [1.56538382e-01 7.11658620e-04 2.14978993e-01 2.61991043e-02
#   2.91832071e-02 2.96098588e-05 5.68943501e-01 3.93983981e-07
#   3.41411517e-03 1.02781621e-06]
#  [2.48365578e-10 1.00000000e+00 2.00976958e-13 7.98070721e-10
#   9.32194211e-10 1.07140482e-28 4.39790954e-10 1.66889169e-18
#   4.66971018e-20 5.03119784e-22]
#  [3.19490960e-06 7.77819551e-08 8.33698511e-02 2.41169146e-05
#   9.12203312e-01 6.46061199e-12 4.39138850e-03 1.02031556e-13
#   8.02668819e-06 5.50963301e-13]
#  [6.97979843e-08 1.18356245e-17 8.98922852e-04 8.37072321e-07
#   1.54577999e-03 1.75743765e-18 9.97554362e-01 2.07102986e-24
#   1.56684776e-09 1.50689479e-20]
#  [3.01208281e-10 9.70095892e-23 6.19716362e-11 2.80427013e-16
#   4.99638464e-17 9.99977708e-01 6.79493517e-13 2.10204416e-05
#   5.37867972e-10 1.36943754e-06]
#  [6.00873448e-11 2.72434621e-16 3.38146720e-13 5.94325961e-16
#   2.94824047e-15 2.72541511e-04 5.59155824e-14 9.99444425e-01
#   1.39812050e-06 2.81551969e-04]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [9 2 1 1 6 1 4 6 5 7]