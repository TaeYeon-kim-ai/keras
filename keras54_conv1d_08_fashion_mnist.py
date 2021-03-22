#fashion_mnist CNN구현


import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#이미지 보기
#plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show(xs


#1.1 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)
x_train=x_train.reshape(x_train.shape[0], 14*14, 4)/255.
x_test=x_test.reshape(x_test.shape[0], 14*14, 4)/255.
x_val=x_val.reshape(x_val.shape[0], 14*14, 4).astype('float32')/255.

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

#tensorflow _to_categorica
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM, Input
model = Sequential()
model.add(Conv1D(filters = 100, kernel_size=(2), strides =1, input_shape = (14*14, 4))) 
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(42, kernel_size=(3)))
model.add(Dropout(0.2))
model.add(Conv1D(40, kernel_size=(3)))
model.add(Dropout(0.2))
model.add(Conv1D(30, kernel_size=(3)))
model.add(Conv1D(30, kernel_size=(3)))
model.add(Conv1D(30, kernel_size=(3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

# model = Sequential()
# model.add(LSTM(100, input_shape = (14*14, 4))) 
# model.add(Dropout(0.25))
# model.add(Dense(42, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(42, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(50, activation= 'relu'))
# model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
# print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

#Conv1d
# loss :  [0.3457416892051697, 0.8755999803543091]
# [9 2 1 1 6 1 4 6 5 7]

#LSTM
# loss :  [0.9185488224029541, 0.6575999855995178]
# [9 2 1 1 6 1 4 6 5 7]

