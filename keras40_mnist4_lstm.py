#X_train.shape = (N, 28, 28) =(N, 28*28) = (N, 764) = input_shape = (764, ) = (28*28 , )
# 주말과제
# dence모델로 구성 input_shape(28*28, )

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

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=55)

print(x_train[0])
print(y_train[0])

print("y_trian[0] : ", y_train[0])
print(x_train[0].shape) #(28, 28)

#1.1 데이터 전처리
#데이터 전처리를 해야함(Min, Max)
x_train=x_train.reshape(x_train.shape[0], 14*7, 8)/255.
x_test=x_test.reshape(x_test.shape[0], 14*7, 8)/255.
x_val=x_val.reshape(x_val.shape[0], 14*7, 8).astype('float32')/255.
# x_train=x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
# x_test=x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.
# x_val=x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])/255.
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.max)
print(x_train.min)
print(x_train.shape, y_train.shape) #(48000, 784) (48000,)
print(x_test.shape, y_test.shape)   #(10000, 784) (10000,)
print(x_val.shape, y_val.shape)   #(12000, 784) (12000,)

#sklearn.onehotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, Input
input1 = Input(shape = (14*7, 8))
lstm = LSTM(400, activation = 'relu', input_shape = (14*7, 8))(input1)
dense1 = Dense(300, activation = 'relu')(lstm)
dense1 = Dense(200, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
outputs = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [0.3718983232975006, 0.8569999933242798]
# [[8.4678513e-09 1.6835662e-04 6.6213892e-05 5.5894679e-03 7.0906367e-07
#   3.7455549e-03 6.6289391e-07 9.8727697e-01 8.5296406e-04 2.2990673e-03]
#  [2.0348642e-04 4.7492977e-02 8.7825114e-01 3.4589201e-02 4.0690917e-05
#   3.0395126e-02 4.3261675e-03 4.2339070e-03 4.0002514e-04 6.7353321e-05]
#  [2.4209994e-05 9.9666494e-01 3.2279513e-05 1.1855762e-04 1.1685338e-03
#   7.7728488e-05 1.3664462e-05 7.7747874e-04 9.8714151e-04 1.3544252e-04]
#  [9.9351513e-01 3.0865900e-05 9.4791967e-04 1.5039188e-04 1.1817697e-04
#   2.7752631e-05 3.9346726e-03 1.0112819e-06 1.2663082e-03 7.5721696e-06]
#  [5.8313116e-08 1.9582145e-05 6.7880697e-05 1.0552794e-06 9.9866903e-01
#   1.3952684e-05 2.5196289e-04 4.3554905e-05 6.9809212e-05 8.6319767e-04]
#  [3.4134737e-06 9.9794489e-01 8.4211670e-06 6.2998253e-05 5.2094157e-04
#   3.0819334e-05 2.1736751e-06 8.4733171e-04 4.8813870e-04 9.0794259e-05]
#  [2.4547850e-04 4.7456929e-03 1.7176284e-02 2.1457323e-03 8.0218911e-01
#   1.1020269e-02 7.0141209e-03 1.1421751e-02 7.1358006e-03 1.3690566e-01]
#  [1.7923135e-06 8.2598394e-04 4.7977449e-04 1.3218345e-03 3.7746798e-02
#   2.0528049e-03 5.0867038e-05 6.9255908e-03 1.5999202e-03 9.4899470e-01]
#  [1.3174234e-04 7.0039337e-03 9.5968360e-01 5.5869366e-03 5.1881530e-04
#   8.5809771e-03 5.3008362e-03 1.2869338e-02 1.4181822e-04 1.8197668e-04]
#  [2.1019990e-07 3.4248582e-04 1.6364713e-04 1.5722954e-03 5.7911011e-03
#   1.6169185e-03 9.2458113e-06 1.3906259e-02 6.3977909e-04 9.7595799e-01]]
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