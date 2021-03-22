#사이킷런 데이터셋
#LSTM으로 모델링
#Dense와 성능비교
#다중분류 #반드시
import numpy as np
from sklearn.datasets import load_iris

#1. 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])
#1.1 전처리 / minmax, train_test_split

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                          #. Set
y = one.transform(y).toarray()      #. transform
print(x.shape) #(150,4)
print(y.shape) #(150,3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) #(150,4, 1)
print(y.shape) #(150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
# input1 = Input(shape=(x.shape[1] ,1))
# conv1d = Conv1D(100, 2, padding='SAME')(input1)
# maxp = MaxPooling1D(2)(conv1d)
# conv1d = Conv1D(100, 2, activation='relu', padding='SAME')(maxp)
# drop = Dropout(0.2)(conv1d)
# fla = Flatten()(drop)
# dense1 = Dense(36, activation='relu')(fla)
# dense1 = Dense(40, activation='relu')(dense1) 
# dense1 = Dense(40, activation='relu')(dense1)
# dense1 = Dense(40, activation='relu')(dense1)
# outputs = Dense(3, activation= 'softmax')(dense1)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(100, activation='relu')(input1)
dense1 =  Dense(100, activation='relu')(lstm)
drop = Dropout(0.2)
dense1 = Dense(36, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1) 
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
outputs = Dense(3, activation= 'softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()


#3. 컴파일,
#다중분류일 경우 : 
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[-5:-1])
print(y_pred)
print(y_train[-5:-1])
print(np.argmax(y_pred, axis = -1))

#Conv1d
# loss :  0.08779489994049072
# acc :  0.9666666388511658

#LSTM
# loss :  0.15984024107456207
# acc :  0.9333333373069763