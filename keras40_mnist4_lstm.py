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
x_train = x_train.reshape(60000, 392, 2).astype('float32')/255.
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
x_test = x_test.reshape(10000, 392, 2)/255.
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.max)
print(x_train.min)

#sklearn.onehotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, Input
input1 = Input(shape = (392, 2))
lstm = LSTM(200, activation = 784, input_shape = (392, 2))
dense1 = Dense(100, activation = 'relu')(lstm)
dense1 = Dense(100, activation = 'relu')(lstm)
dense1 = Dense(100, activation = 'relu')(lstm)
dense1 = Dense(100, activation = 'relu')(lstm)
dense1 = Dense(100, activation = 'relu')(lstm)
dense1 = Dense(100, activation = 'relu')(lstm)
outputs = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

