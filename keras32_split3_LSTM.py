# 과제 및 실습
#LSTM으로 구성
# 전처리, early_stopping, MinMax, 등
# 데이터 1~ 100 / 6개씩 잘라라
#       x                y 
# 1, 2, 3, 4, 5          6
# ... 
# 95 , 96, 97, 98, 99     100

# predict를 만들것
# 96, 97, 98, 99, 100 -> 101
# ...
#100, 101, 102, 103, 104, 105
#예상 predict는 (101, 102, 103, 104, 105)
import numpy as np

#1. 데이터
a = np.array(range(1, 101))
size = 6

def split_x(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size)]  
        aaa.append(subset) 
    print(type(aaa)) 
    return np.array(aaa) 
dataset = split_x(a, size) 
#print(dataset) #(1~100)
x = dataset[:, :5]
y = dataset[:, -1]

#======================================================#
b = np.array(range(96, 106))
size2 = 6
dataset_pred = split_x(b, size2) 
x_predict = dataset_pred[:, :5]
y_predict = dataset_pred[ :, -1]
print(y_predict)
print(x_predict)

#y = y.reshape(y.shape[0], y.shape[1], 1)
# print(x.shape) #(95, 5, 1)
# print(y.shape) #(95, 1, 1)
# print(x.shape) # (95, 5)
# print(y.shape) # (95,)

#1.1데이터 정제(MinMax)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 50)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 50)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_tranin = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x.val.shape[1], 1)

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
input1 = Input(shape = (5,1))
lstm1 = LSTM(100, activation = 'relu')(input1)
dense1 = Dense(100, activation = 'relu')(lstm1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(50)(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_test, y_test, epochs = 1000, batch_size = 1, validation_data = (x_val, y_val), callbacks = [early_stopping])

#평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
# x_pred = x_predict.reshape(5,5,1)
result = model.predict(x_pred)
print('result : ', result)
