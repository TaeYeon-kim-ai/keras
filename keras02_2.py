# 네이밍 룰
# 자바  카멜케이스 : keras02Numpy.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras import models
#from tensorflow import keras
from tensorflow.keras.layers import Dense

# 1. 데이터 셋
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])
x_test = np.array(list(range(101,111))) #비교 data
y_test = np.array(list(range(111,121)))
x_predict = np.array([111,112,113])

# 2.모델
model = Sequential()
model.add(Dense(7, input_dim = 1, activation= 'relu'))
model.add(Dense(1))
#model = models.Sequential()
#model = keras.models.Sequential()
#model.add(Dense(5, input_dim = 1, activation= 'relu'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=330, batch_size=21)

# 4. 평가
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result = model.predict([x_predict])
print('result : ', result)

