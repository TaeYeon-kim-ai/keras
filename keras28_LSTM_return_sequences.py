#keras23_3을 카피해서 
#LSTM층을 두개 만들 것
# ex)
# model.add(LSTM(10, input_shape = (3,1)))
# model.add(LSTM(10)

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

#x = x.reshape(13,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1) #(13, 3, 1)
              # 행   ,    열     ,  다음값

#2. 모델구성(LSTM)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(13, return_sequences = True, activation = 'linear', input_shape = (3,1)))
model.add(LSTM(26, activation = 'linear')) #LSTM 2회 사용 시  return_sequences=True 추가
model.add(Dense(52))
model.add(Dense(13))
model.add(Dense(13))
model.add(Dense(1))
model.summary()




"""
#3.컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping 
# early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x, y, epochs = 200, batch_size = 1)

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

# loss :  [0.009817657992243767, 0.0] LSTM 1층
# result :  [[80.22657]]

# loss :  [0.04210755601525307, 0.0] LSTM 2층
# result :  [[80.84903]]

# loss :  [0.020084165036678314, 0.0] LSTM 3층 (1)
# result :  [[81.62996]]

'''
LSTM을 여러번 사용할 경우 hidden layer에 연속적인 시계열 데이터를 던져주지 않음 
하여 성능이 더 좋게나오지 않을 가능성이 높음(데이터에따라 다름)

가. LSTM return_sequences 적용 시  
ㅇ LSTM은 3차원을 받아들이지만 출력은 Dense와 동일한 2차원으로 함
ㅇ but, LSTM을 2개 엮을 경우 2번째 LSTM2에 input shape가 3차원으로 들어가야 하므로      
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 13)             780
_________________________________________________________________
lstm_1 (LSTM)                (None, 26)                4160
_________________________________________________________________
dense (Dense)                (None, 52)                1404
_________________________________________________________________
dense_1 (Dense)              (None, 13)                689
_________________________________________________________________
dense_2 (Dense)              (None, 13)                182
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 14
=================================================================
 - layer1 의 out node의 수 는 Output Shape1의 가장끝에 위치함

4 ( 1, 13, 1) 13    = 780
4 * 40* 26    	= 4160
-- 840구하기 

'''
"""