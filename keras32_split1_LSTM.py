# 모델을 구성하시오
import numpy as np

#1. 데이터
a = np.array(range(1, 11))
size = 5

def split_x(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1 ): #for반복문 i를 반복해라 size + 1 까지
        subset = seq[i : (i + size)]  #seq를 구성해라 i(1)부터 i+size(5)까지
        aaa.append(subset) # aaa에 추가해라 [] 한바퀴돌
    print(type(aaa)) #aaa 의 타입을 추가해라
    return np.array(aaa) #aaa를 반환하라

dataset = split_x(a, size) #dataset에 추가
print("===========================")
#print(dataset) #split_x datasets을 0~10까지 size 5까지 순서대로 넣기

x = dataset[:,0:4]  # [0:6, 0:4] [:,0:4] or [:,0:-2]
y = dataset[:,4:] # p[0:6, 4] [:,4]or [:,-1]
#print(x.shape) #  (6, 4)
#print(y.shape) #  (6, 1)
print(x) 
print(y) # [ 5  6  7  8  9 10]

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)
print(x.shape) # (6, 4, 1)
print(y.shape) # (6, 1, 1)

#2. 모델 구성(LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (4, 1)))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))
model.summary()

#3. 컴파일
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 50, batch_size = 1)

#평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

#x_pred = np.array([7, 8, 9, 10])
x_pred = dataset[-1,1:] # [ 7  8  9 10] 1뒤로부터 끝가지 자르기
print('x_pred : ', x_pred)
x_pred = x_pred.reshape(1,4,1)
result = model.predict(x_pred)
print('result : ', result)

# model.add(LSTM(10, activation = 'relu', input_shape = (4, 1)))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(30))
# model.add(Dense(1))
# loss :  [0.003162645734846592, 0.0]
# [ 7  8  9 10]
# result :  [[10.99304]]
