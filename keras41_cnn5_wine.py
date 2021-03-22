import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target
# print(x)
# print(y)
print(x.shape) # (178, 13)
print(y.shape) # (178,)

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                          #. Set
y = one.transform(y).toarray()      #. transform
# print(y)
print(x.shape) #(178, 13)
print(y.shape) #(178,3)

x = x.reshape(x.shape[0], x.shape[1], 1, 1) #(178, 13, 1, 1)
y = y.reshape(y.shape[0], y.shape[1], 1, 1) #(178, 3, 1, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (13, 1, 1))) 
model.add(Conv2D(500, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(200, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))
model.summary()

#3. 컴파일,
#다중분류일 경우 : 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[-5:-1])
print(y_pred)
print(y_train[-5:-1])

#y값 중에서 가장 큰 값을 1로 바꾼다 : argmax
#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = -1))
#print(np.argmax(y_pred, axis = 2)) 

# loss :  1.943482129718177e-05
# acc :  1.0
# [[1.0000000e+00 2.2382447e-09 1.4489170e-09]
#  [5.6554582e-06 9.9999440e-01 3.0707078e-08]
#  [6.3094944e-08 3.6121787e-07 9.9999952e-01]
#  [4.6856638e-07 9.9999940e-01 1.3655827e-07]]
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]
# [0 1 2 1]

