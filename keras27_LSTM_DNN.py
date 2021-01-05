import numpy as np
#코딩해서 80 출력

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10], 
              [9,10,11], [10,11,12], [20,30,40], 
              [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred=np.array([50,60,70])

print("x : shape", x.shape) # (13, 3)
print("y : shape", y.shape) # (13,)

#1.1 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#MinMaxScaler 필수로 쓸것
scaler = MinMaxScaler()
scaler.fit(x)
x_train = scaler.transform(x)#x_train에 trans

# x = x.reshape(13,3,1)
x_pred=x_pred.reshape(1,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 2)

#2. 모델구성(DNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(Dense(13, activation = 'linear', input_shape = (3,)))
model.add(Dense(26))
model.add(Dense(52))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))
model.summary()

#3.컴파일, 훈련
#EarlyStopping 사용, validation_data사용
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 1, verbose = 1, validation_data = (x_val, y_val),callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_train,y_train)
print('loss : ', loss)

y_predict = model.predict(x_pred)
print('result : ',y_predict)

# loss :  [0.03231086581945419, 0.0]
# result :  [[79.23537]]

# loss :  [0.004554993938654661, 0.0]
# result :  [[80.282875]]

# loss :  [0.1634834110736847, 0.0] LSTM_seq
# result :  [[80.04349]]

# loss :  [0.7082670331001282, 0.0] DNN early_stopping 132 / patience = 30
# result :  [[80.64098]]