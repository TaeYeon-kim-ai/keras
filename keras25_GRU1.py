#GRU

import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # (4,3)
y = np.array([4,5,6,7]) # (4, )

print("x : shape", x.shape) #(4,3)
print("y : shape", y.shape) #(4,)

#1개씩 잘라서 작업을 하기 위해 shape를 바꿈

x = x.reshape(4,3,1) #LSTM쓰기위한 shape 3D작업 [[1],[2],[3]],[[2],[3],[4]]...... 반드시 '3D'
             #(행,열,몇개씩 자르는지?)

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
model = Sequential()
model.add(GRU(10, activation = 'relu', input_shape = (3,1))) #3D 받아들임 행, 열, + 어떻게자를건지 param 4 * (n + 10 + 1) * 10
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 10)                390            = GRU =  3 * (n + m + 1) * m  where 30??
_________________________________________________________________              3 * (1 +1 0+ 1) *10  = 360 ????
dense (Dense)                (None, 20)                220                     default activation = tanh
_________________________________________________________________              input node에 3배도 더해주나?
dense_1 (Dense)              (None, 10)                210
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 831
Trainable params: 831
Non-trainable params: 0
'''


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100,  batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

x_pred = np.array([5,6,7]) # (3,) -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print('result : ', result)


# loss :  0.005356112495064735 LSTM
# result :  [[7.8571596]]

# loss :  0.003049550112336874   GRU
# result :  [[7.7128334]]












