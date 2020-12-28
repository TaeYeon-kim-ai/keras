import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# y=wx + b
# 1. 데이터 셋   
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8]) #비교 data
y_test = np.array([6,7,8])

# 2.모델
model = Sequential()
model.add(Dense(5, input_dim = 1, activation= 'relu'))
model.add(Dense(3)) #안쓰면 디폴트 값 있음. linear 
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])#accuracy = 정확도
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])#MSE (Mean Squared Error)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])#MAE (Mean Absolute Error)
model.fit(x_train, y_train, epochs=100, batch_size=6)

# 4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

#result = model.predict([9])
result = model.predict(x_train)
print('result : ', result)

