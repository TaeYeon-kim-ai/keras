# 인공지능 계의  hellow world라 불리는 mnist
# 실습 완성하시오


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

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
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
#(정수형 -> float 타입으로 바꾸기) 전처리 (실수)255. : 0~1사이로 변환
x_test = x_test.reshape(10000, 28, 28, 1)/255.
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
model.add(Conv2D(32, (2,2), padding = 'SAME'))
model.add(Conv2D(32, (2,2), padding = 'SAME'))
model.add(Conv2D(32, (2,2), padding = 'SAME'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size= 2))

model.add(Flatten())
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'softmax'))
model.add(Dense(10))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [early_stopping])
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[:10])
print(y_test[:10])
print(y_pred)

# loss :  6.342271327972412
# acc :  0.0982000008225441

#(실습)완성하시오
#지표는 acc // 0.985 이상
#(응용) y_test 10개와 y_pred 10개를 출력하시오
# y_test[:10] = (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10] = (?,?,?,?,?,?,?,?,?,?)
