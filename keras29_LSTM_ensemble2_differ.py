#실습: 2개의 모델을 하나는 LSTM, 하나는 Dense로
# 29_1 번과 성능비교
# predict = 85근사치
import numpy as np
from numpy import array

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
              [9,10,11], [10,11,12], [20,30,40], 
              [30,40,50], [40,50,60]])

x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], 
              [50,60,70], [60,70,80], [70,80,90], [80,90,100], 
              [90,100,110], [100,110,120], [2,3,4], 
              [3,4,5], [4,5,6]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75]) #(3,) -> (1,3) -> (1, 3, 1)
x2_predict = array([65,75,85]) #(3,) -> (1,3) -> (1, 3, 1)

print("x : shape", x1.shape) #(13,3)
print("y : shape", x2.shape) #(13,3)
print("y : shape", y.shape) #(13,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)

x1 = x1.reshape(13,3,1)
x2 = x2.reshape(13,3,1)

#1. 데이터 전처리
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, shuffle = False, train_size = 0.8)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

#모델1 
input1 = Input(shape = (3,1))
lstm1 = LSTM(100, activation = 'linear', input_shape = (3,1))(input1)
dense1 = Dense(50, activation = 'linear')(lstm1)
dense1 = Dense(10, activation = 'linear')(dense1)
dense1 = Dense(10, activation = 'linear')(dense1)
outputs2 = Dense(1)(dense1)

#모델2
input2 = Input(shape = (3,))
#lstm2 = LSTM(10, activation = 'linear', input_shape = (3,1))(input2)
dense2 = Dense(10, activation = 'linear')(input2)
dense2 = Dense(30, activation = 'linear')(dense2)
dense2 = Dense(10, activation = 'linear')(dense2)
dense2 = Dense(10, activation = 'linear')(dense2)
outputs1 = Dense(1)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle = Dense(10)(merge1)
middle = Dense(10)(middle)
middle = Dense(10)(middle)

#모델 분리
output1 = Dense(10)(middle)
output1 = Dense(50)(output1)
output1 = Dense(1)(output1)

#모델 선언
model = Model(inputs = [input1, input2], outputs = output1) #input2개 이상은..
model.summary()


#컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit([x1_test, x2_test], y_test, epochs=100, batch_size=1, validation_split=0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

x1_pred = x1_predict.reshape(1, 3, 1)
x2_pred = x2_predict.reshape(1, 3, 1)
result = model.predict([x1_pred, x2_pred])
print('result: ', result)

# loss :  [0.042945101857185364, 0.0] LSTM
# result:  [[84.855995]]

# loss :  [0.007720783818513155, 0.0] LSTM, DNN ensemble
# result:  [[188.68988]]
































