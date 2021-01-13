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

# print(y)
print(x.shape) #(178, 13)
print(y.shape) #(178,3)

y = y.reshape(-1,1)                 #. y_train => 2D

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_test.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
one.fit(y_train)               
y_train = one.transform(y_train).toarray()     
y_test = one.transform(y_test).toarray()      
y_val = one.transform(y_val).toarray()   

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv1D(filters = 100, kernel_size=(2),padding = 'SAME', input_shape = (13, 1))) 
model.add(Conv1D(500, kernel_size=(2), padding = 'SAME'))
model.add(Conv1D(200, kernel_size=(2), padding = 'SAME'))
model.add(Conv1D(100, kernel_size=(2), padding = 'SAME'))
model.add(Conv1D(100, kernel_size=(2), padding = 'SAME'))
model.add(Conv1D(100, kernel_size=(2), padding = 'SAME'))
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

# loss :  0.33169251680374146
# acc :  0.8611111044883728
# [[8.5227388e-01 7.2438195e-02 7.5287901e-02]
#  [1.3748792e-03 7.7232337e-01 2.2630176e-01]
#  [2.9319166e-03 1.4299753e-01 8.5407054e-01]
#  [2.8391554e-05 9.9926752e-01 7.0417696e-04]]
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]
# [0 1 2 1]


