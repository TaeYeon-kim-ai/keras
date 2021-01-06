# 과제 및 실습
#LSTM으로 구성
# 전처리, early_stopping, MinMax
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
b = np.array(range(96,106))
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
dataset_pred = np.array(split_x(b, 6))  
x_pred = dataset_pred[:, :5]
y_pred = dataset_pred[ :,-1]
print(x_pred)
print(y_pred)

#y = y.reshape(y.shape[0], y.shape[1], 1)
# print(x.shape) #(95, 5, 1)
# print(y.shape) #(95, 1, 1)
# print(x.shape) # (95, 5)
# print(y.shape) # (95,)

#1.1데이터 정제(MinMax)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2,shuffle = True, random_state = 50)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8,  test_size = 0.2, shuffle = True, random_state = 50)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_tranin = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
x_val = scaler.transform(x_val)
#x_pred = scaler.transform(x_pred)

print(x_train.shape) #(60,5)
print(x_test.shape)  #(19,5)
print(x_val.shape)   #(16,5)
print(x_pred.shape)  #(5,5)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) #(95,5) => (95,5,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1) #(95,5) => (95,5,1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1) #(95,5) => (95,5,1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1) #(5,5) => (5,5,1)

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
input1 = Input(shape = (5,1))
lstm1 = LSTM(100, activation = 'relu', input_shape = (5,1))(input1)
dense1 = Dense(70, activation = 'linear')(lstm1)
dense1 = Dense(60, activation = 'linear')(dense1)
dense1 = Dense(50, activation = 'linear')(dense1)
dense1 = Dense(50, activation = 'linear')(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 5, validation_data = (x_val, y_val), callbacks = [early_stopping])

#평가, 예측
loss = model.evaluate(x_train, y_train)
print("loss : ", loss)
result = model.predict(x_pred)
print('result : ', result)

# loss :  [0.00027045546448789537, 0.0] LSTM   earlystopping 223 // patience 30
# result :  [[101.0326  ]
#  [102.034744]
#  [103.03702 ]
#  [104.03941 ]
#  [105.04197 ]]

