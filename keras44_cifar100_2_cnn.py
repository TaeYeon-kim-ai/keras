import numpy as np

#1. 데이터
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

#1.1 데이터 전처리
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])/225.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])/225.
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val=train_test_split(x_train, y_train, train_size=0.8, random_state=66)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3])/225.

print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape) 
# (40000, 32, 32, 3)
# (10000, 32, 32, 3)
# (10000, 32, 32, 3)

#tensorflow _to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (32, 32, 3))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(200, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Flatten())
model.add(Dense(200, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))
