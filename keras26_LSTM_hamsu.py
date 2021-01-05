#keras23_LSTM3_scale을 함수로 코딩
#결과치 비교

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

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
input1 = Input(shape = (3,1))
lstm1 = LSTM(13, activation = 'linear', input_shape = (3,1))(input1)
dense1 = Dense(26)(lstm1)
dense2 = Dense(52)(dense1)
dense3 = Dense(13)(dense2)
dense4 = Dense(13)(dense3)
outputs = Dense(1)(dense4)
model = Model(inputs = input1, outputs = outputs)
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

# loss :  [0.1634834110736847, 0.0] LSTM_Seq
# result :  [[80.04349]]

# loss :  [0.010905833914875984, 0.0] LSTM_hamsu   patience = 30 / 121
# result :  [[80.81497]]