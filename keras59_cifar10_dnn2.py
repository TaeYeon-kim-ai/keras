#(n , 32, 32, 3) -> (n, 10)
import numpy as np

#1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y_train = y_train.reshape(-1,1)                 #. y_train => 2D
one.fit(y_train)                          #. Set
y_train = one.transform(y_train).toarray()      #. transform
y_test = one.transform(y_test).toarray()      #. transform
# print(y)

#1.1 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=1000)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3])

x_train = x_train.reshape(x_train.shape[0], 16*48*4)
x_test = x_test.reshape(x_test.shape[0], 16*48*4)
x_val = x_val.reshape(x_val.shape[0], 16*48*4)
print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape) 
# (40000, 3072)
# (10000, 3072)
# (10000, 3072)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_tranin = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


print(y_train.shape) 
print(x_train.shape)
# (40000, 10)
# (40000, 3072)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
input1 = Input(shape=(16*48*4,))
dense1 = Dense(72, activation='relu', input_shape = (16*48*4,))(input1)
dense1 = Dense(64, activation= 'relu')(dense1)
dense1 = Dense(64, activation= 'relu')(dense1)
dense1 = Dense(64, activation= 'relu')(dense1)
dense1 = Dense(32, activation= 'relu')(dense1)
dense1 = Dense(32, activation= 'relu')(dense1)
dense1 = Dense(16, activation= 'relu')(dense1)
outputs = Dense(10, activation='softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 30, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [3.007906913757324, 0.10000000149011612]
# [3 8 8 0 6 6 1 6 3 1]
# loss :  [9.670855522155762, 0.10000000149011612]
# [3 8 8 0 6 6 1 6 3 1]