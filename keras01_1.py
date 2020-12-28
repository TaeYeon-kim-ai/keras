import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성(인공신경망)
model = Sequential() #순차적 모델
model.add(Dense(5, input_dim=1, activation='linear')) #dam 1개 차원
model.add(Dense(3, activation='linear')) # 모델 수 hidden layer
model.add(Dense(4))# 모델 수 hidden layer
model.add(Dense(1))# 모델 수 hidden layer

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc']) 
model.fit(x, y, epochs=1000, batch_size=1) 

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)
print('acc : ', acc) 

x_pred = np.array([4])
#result = model.predict([x])
#result = model.predict([4])
result = model.predict([x_pred]) #predict에 넣는 값은 예측 하고 싶은 값을 넣는다. 비교군 데이터
print('result : ', result)