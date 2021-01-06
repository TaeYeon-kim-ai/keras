

import numpy as np
from numpy import array

#1. 데이터
x1 = np.array([[1,2], [2,3], [3,4], [4,5], 
              [5,6], [6,7], [7,8], [8,9], 
              [9,10], [10,11], [20,30], 
              [30,40], [40,50]])#(13,2)

x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], 
              [50,60,70], [60,70,80], [70,80,90], [80,90,100], 
              [90,100,110], [100,110,120], [2,3,4], 
              [3,4,5], [4,5,6]])#(13, 3)

y1 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], 
              [50,60,70], [60,70,80], [70,80,90], [80,90,100], 
              [90,100,110], [100,110,120], [2,3,4], 
              [3,4,5], [4,5,6]])#(13,2)
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])#(13,)

x1_predict = array([55,65]) #(2,) -> (1,2) -> (1, 2, 1) 3개
x2_predict = array([65,75,85]) #(3,) -> (1,3) -> (1, 3, 1) 1개

print("x1 : shape", x1.shape) #(13,2)
print("x2 : shape", x2.shape) #(13,3)
print("y1 : shape", y1.shape) #(13,3)
print("y2 : shape", y2.shape) #(13,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 2, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
# y1 = y1.reshape(y1.shape[0], y1.shape[1], 1) #(13, 2, 1)
# y2 = y2.reshape(y2.shape[0], y2.shape[1], 1) #(13, 3, 1)

#1. 데이터 전처리
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle = True, train_size = 0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle = True, train_size = 0.8)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

#모델1 
input1 = Input(shape = (2,1))
lstm1 = LSTM(10, activation = 'relu')(input1)
dense1 = Dense(200)(lstm1)
dense1 = Dense(10)(dense1)
dense1 = Dense(10)(dense1)
outputs1 = Dense(2)(dense1)

#모델2
input2 = Input(shape = (3,1))
lstm2 = LSTM(10, activation = 'relu')(input2)
dense2 = Dense(200)(lstm2)
dense2 = Dense(50)(dense2)
dense2 = Dense(10)(dense2)
dense2 = Dense(10)(dense2)
outputs2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(40)(merge1)
middle1 = Dense(30)(middle1)
middle1 = Dense(10)(middle1)

#모델 분기1
output1 = Dense(20)(middle1)
output1 = Dense(10)(output1)
output1 = Dense(3)(output1)

#모델 분기2
output2 = Dense(20)(middle1)
output2 = Dense(10)(output2)
output2 = Dense(1)(output2)

#모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2]) #input2개 이상은..
model.summary()


#컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit([x1_test, x2_test], [y1_test, y2_test], epochs=100, batch_size=1, validation_split=0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ', loss)

x1_pred = x1_predict.reshape(1, 2, 1)
x2_pred = x2_predict.reshape(1, 3, 1)
result1 = model.predict([x1_pred, x2_pred])
print('result1 : ', result1)

# loss :  [137.86131286621094, 137.34181213378906, 0.5194948315620422, 1.0, 0.0]
# result1 :  [array([[43.731857, 62.147583, 70.31219 ]], dtype=float32), array([[17.113935]], dtype=float32)]

