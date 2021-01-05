#GRU적용

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
model = Sequential()
model.add(GRU(13, activation = 'linear', input_shape = (3,1)))
model.add(Dense(26))
model.add(Dense(52))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))
model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 13)                624           = GRU =  3 * (n + m + 1) * m  where 30??
_________________________________________________________________             3 * (1 + 13 + 1) * 13  = 360 ????
dense (Dense)                (None, 26)                364                    default activation = tanh
_________________________________________________________________             input node에 3배도 더해주나?
dense_1 (Dense)              (None, 52)                1404
_________________________________________________________________
dense_2 (Dense)              (None, 13)                689
_________________________________________________________________
dense_3 (Dense)              (None, 13)                182
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 14
=================================================================
Total params: 3,277
Trainable params: 3,277
Non-trainable params: 0
'''
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

# loss :  [0.1634834110736847, 0.0] LSTM
# result :  [[80.04349]]

# loss :  [0.8067426085472107, 0.0] GRU
# result :  [[82.480354]]