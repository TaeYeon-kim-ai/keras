#사이킷런
#LSTM으로 모델링
#Dense와 성능비교
#보스턴 집값
#실습:모델구성
import numpy as np
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,   )
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (506, 13, 1)


#1_2. 데이터 전처리(MinMaxScalar)
#ex 0~711 = 최댓값으로 나눈다  0~711/711
# X - 최소값 / 최대값 - 최소값
print("===================")
print(x[:5]) # 0~4
print(y[:10]) 
print(np.max(x), np.min(x)) # max값 min값
print(dataset.feature_names)
#print(dataset.DESCR) #묘사

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)
print(x_train.shape)
print(y_train.shape)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_tranin = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
# x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
# x_val = scaler.transform(x_val) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
# input1 = Input(shape=(x.shape[1] ,1))
# conv1d = Conv1D(50, 2, activation='relu')(input1)
# dense1 = MaxPooling1D(pool_size=1)(conv1d)
# conv1d = Conv1D(50, 2, activation='relu')(dense1)
# dense1 = Dense(36, activation='relu')(dense1)
# dense2 = Dense(40, activation='relu')(dense1) 
# dense3 = Dense(40, activation='relu')(dense2)
# dense4 = Dense(40, activation='relu')(dense3)
# outputs = Dense(1)(dense4)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(50, activation='relu')(input1)
dense1 =  Dense(50, activation='relu')(lstm)
dense1 = Dense(36, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1) 
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
outputs = Dense(1)(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=6, validation_data= (x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) 
print('result :' ,y_predict[0])

#Conv1D
# loss :  61.70733642578125
# mae :  5.866117000579834

#LSTM
# loss :  23.057985305786133
# mae :  3.220595598220825