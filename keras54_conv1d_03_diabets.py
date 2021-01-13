#사이킷런 데이터셋
#LSTM으로 모델링
#회귀모델

import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target



#1_2. 데이터 전처리


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(x.shape[0], x.shape[1], 1) # (442, 10, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_test, x_val, y_test, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle = True, random_state = 66, )

print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

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
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_train, y_train)
print("loss : ", loss)
print("mae : ", mae)
result = np.transpose(y)
y_predict = model.predict(result)

#Conv1D
# loss :  5346.369140625
# mae :  62.078125

#LSTM
# loss :  2481.505126953125
# mae :  40.03818130493164
