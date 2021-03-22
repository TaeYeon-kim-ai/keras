#keras23_3을  카피헤서 Conv1D로 완성하기

import numpy as np
#코딩해서 80 출력

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
              [9,10,11], [10,11,12], [20,30,40], 
              [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x : shape", x.shape) # (13, 3)
print("y : shape", y.shape) # (13,)

x = x.reshape(13,3,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_tset = scaler.transform(x_test)
# x_val = scaler.transform(x_val)

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
model = Sequential()
model.add(Conv1D(filters = 100, kernel_size=(2,), activation='relu' ,input_shape = (3,1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(52, kernel_size=2))
model.add(Flatten())
model.add(Conv1D(52, kernel_size=2))
model.add(Conv1D(52, kernel_size=2))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 1000, batch_size = 1, callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

x_pred = np.array([50,60,70])
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print('result : ', result)

# loss :  [0.03231086581945419, 0.0]
# result :  [[79.23537]]

# loss :  [0.004554993938654661, 0.0]
# result :  [[80.282875]]

# loss :  [0.1634834110736847, 0.0]
# result :  [[80.04349]]