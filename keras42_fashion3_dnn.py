#fashion_mnist CNN구현


import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])/255.

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
model.add(Dense(500, activation='relu', input_shape = (784,)))
model.add(Dropout(0.2))
model.add(Dense(500, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(400, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [0.3507075309753418, 0.8906999826431274]  - DNN
# [[1.0114068e-17 2.8370056e-16 1.9433930e-21 2.4569051e-20 6.2587257e-27
#   1.1099346e-05 9.5151392e-19 2.7659858e-04 1.3660285e-10 9.9971229e-01]
#  [7.5298658e-06 1.2290736e-10 9.9889189e-01 4.1509889e-08 5.5440248e-04
#   1.8987762e-16 5.4626522e-04 1.9420995e-16 1.0209310e-08 6.1435144e-22]
#  [0.0000000e+00 1.0000000e+00 0.0000000e+00 3.5453664e-35 0.0000000e+00
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [4.0117337e-33 1.0000000e+00 0.0000000e+00 1.0063517e-24 5.5420076e-37
#   0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [2.4376409e-01 9.1355229e-05 7.9851849e-03 6.0835048e-03 1.0996484e-03
#   7.4612707e-08 7.4081343e-01 8.1748039e-09 1.6279945e-04 7.1892353e-10]
#  [1.1426930e-24 1.0000000e+00 0.0000000e+00 1.6871981e-19 9.8789830e-29
#   0.0000000e+00 1.2877564e-33 0.0000000e+00 0.0000000e+00 0.0000000e+00]
#  [1.6438934e-18 1.8194599e-23 2.2711747e-05 1.4155446e-14 9.9997580e-01
#   0.0000000e+00 1.4359728e-06 0.0000000e+00 3.9369272e-18 0.0000000e+00]
#  [5.5690493e-06 5.0220550e-10 1.1143013e-03 3.8690150e-06 4.2358663e-02
#   1.4031910e-12 9.5651764e-01 5.9996963e-14 1.5778248e-08 2.5949887e-17]
#  [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
#   1.0000000e+00 0.0000000e+00 2.7530084e-32 0.0000000e+00 2.4559779e-23]
#  [3.1798879e-16 6.0932973e-19 3.4107622e-19 8.9818611e-15 9.7850529e-20
#   1.4642366e-07 1.0200559e-14 9.9999762e-01 2.5155817e-11 2.2999996e-06]]
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