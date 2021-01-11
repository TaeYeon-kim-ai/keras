#X_train.shape = (N, 28, 28) =(N, 28*28) = (N, 764) = input_shape = (764, ) = (28*28 , )
#(N, 28, 28) CNN
#(N, 764) RNN
#(28*28, ) LSTM 
#(28*14,2) LSTM 
#(28*7,4) LSTM 
#(7*7,16) LSTM 
# 주말과제
# dence모델로 구성 input_shape(28*28, )
#boston ,diabets, 
# 인공지능 계의  hellow world라 불리는 mnist

#(실습)완성하시오
#지표는 acc // 0.985 이상
#(응용) y_test 10개와 y_pred 10개를 출력하시오
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #1생략 흑백
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,) #1생략 흑백

print(x_train[0])
print(y_train[0])

print("y_trian[0] : ", y_train[0])
print(x_train[0].shape) #(28, 28)

#1.1 데이터 전처리
#데이터 전처리를 해야함(Min, Max)
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=55)

x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.
x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])/255.

print(x_train.max)
print(x_train.min)

#tensorflow.keras .. to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(500, activation= 'relu', input_shape = (784,)))
model.add(Dense(400, activation= 'relu')) 
model.add(Dense(300, activation= 'relu'))
model.add(Dense(250, activation= 'relu'))
model.add(Dense(250, activation= 'relu'))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [0.09116600453853607, 0.9779000282287598]
# [[2.05746275e-07 2.79304527e-06 5.79399239e-05 2.86863475e-08
#   5.19158675e-05 9.56795798e-09 8.14814327e-10 9.99707878e-01
#   3.44381328e-06 1.75779074e-04]
#  [1.93170528e-08 6.93986672e-15 1.00000000e+00 1.32798017e-11
#   7.29948938e-15 5.65265201e-14 1.25495969e-09 7.63613738e-12
#   1.41629668e-11 4.14613615e-14]
#  [8.59525784e-11 9.99999642e-01 9.40291329e-08 1.03130590e-12
#   1.39162654e-10 6.48883239e-11 7.28771568e-08 1.40417927e-10
#   8.95537937e-08 8.75773900e-13]
#  [9.99918580e-01 1.99210631e-11 2.06170262e-06 2.97419049e-08
#   4.87801628e-07 2.78386942e-05 3.21447333e-05 1.62675597e-08
#   1.08784834e-05 7.97651228e-06]
#  [1.13525367e-07 7.88041348e-08 1.76674746e-06 4.00015481e-12
#   9.99803603e-01 5.64378162e-08 3.30965995e-05 6.41564895e-07
#   9.66545599e-08 1.60692158e-04]
#  [3.80808995e-10 9.99999404e-01 2.43158382e-07 7.37325305e-12
#   6.05243633e-10 2.61428573e-10 1.61048362e-07 9.80926340e-10
#   2.79251310e-07 7.15098293e-12]
#  [2.38717166e-05 2.39372566e-05 1.69644845e-04 4.00890201e-08
#   9.95679975e-01 1.61729258e-05 5.92472323e-04 8.92623721e-05
#   4.16788935e-05 3.36302142e-03]
#  [3.88256938e-09 5.57031354e-09 1.27730604e-09 1.51428466e-07
#   7.20426178e-05 2.11959062e-07 4.63511972e-11 8.42771271e-07
#   9.10794711e-08 9.99926686e-01]
#  [8.49335978e-04 9.27423476e-04 4.62811149e-04 1.05891761e-03
#   3.99533752e-03 9.19198096e-01 5.88678606e-02 1.61656470e-04
#   1.20783160e-02 2.40024948e-03]
#  [7.80996416e-13 2.42627337e-12 1.96749504e-13 1.04459955e-10
#   1.40269094e-06 2.48899179e-10 1.64700284e-15 3.09528692e-09
#   5.48854955e-11 9.99998569e-01]]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [7 2 1 0 4 1 4 9 5 9]