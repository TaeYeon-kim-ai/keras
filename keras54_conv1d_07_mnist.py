import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) =  fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #1생략 흑백
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,) #1생략 흑백

print(x_train[0])
print(y_train[0])

print("y_trian[0] : ", y_train[0])
print(x_train[0].shape) #(28, 28)

# #이미지 보기
# #plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()

#1.1 데이터 전처리
#데이터 전처리를 해야함(Min, Max)
x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
x_test = x_test.reshape(10000, 28, 28)/255.
#x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.max)
print(x_train.min)

#sklearn.onehotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
#                         train_size = 0.8, shuffle = True, random_state = 66)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, LSTM

# model = Sequential()
# model.add(Conv1D(filters = 100, kernel_size=(2), strides =1, input_shape = (28, 28))) 
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.25))
# model.add(Conv1D(42, kernel_size=(3)))
# model.add(Dropout(0.2))
# model.add(Conv1D(40, kernel_size=(3)))
# model.add(Dropout(0.2))
# model.add(Conv1D(30, kernel_size=(3)))
# model.add(Conv1D(30, kernel_size=(3)))
# model.add(Conv1D(30, kernel_size=(3)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(50, activation= 'relu'))
# model.add(Dense(10, activation= 'softmax'))

model = Sequential()
model.add(LSTM(100, input_shape = (28, 28))) 
model.add(Dropout(0.25))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(42, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dense(42, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [0.0957411602139473, 0.975600004196167] Conv2D
# [7 2 1 0 4 1 4 9 5 9]

# loss :  [0.44213730096817017, 0.8378000259399414] LSTM
# [9 2 1 1 6 1 4 6 5 7]
