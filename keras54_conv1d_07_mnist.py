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
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(200, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(200, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(150, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
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

# loss :  [0.0957411602139473, 0.975600004196167]
# [[2.3473370e-13 3.5473342e-08 3.1223895e-08 4.7982365e-07 1.3058732e-11
#   5.6865125e-12 1.0871746e-17 9.9999940e-01 7.8487838e-09 2.2004231e-08]
#  [1.0515940e-08 6.0934480e-07 9.9999917e-01 8.6361354e-08 1.9161255e-09
#   3.6792288e-11 3.3695113e-09 1.2978587e-07 2.9971328e-10 2.2728156e-11]
#  [1.1606807e-09 9.9994993e-01 2.2005126e-07 6.2554996e-08 3.8541661e-08
#   1.3250397e-07 5.9824634e-07 3.1741067e-05 1.7324681e-05 2.1037515e-08]
#  [9.9757534e-01 1.6300210e-07 7.8601041e-04 4.8423681e-05 1.2471904e-04
#   2.4472465e-04 3.7388990e-04 2.0552457e-04 1.7172794e-04 4.6947852e-04]
#  [1.6636349e-04 1.5609086e-02 1.5034185e-02 2.1073311e-05 9.5494813e-01
#   1.4350747e-04 1.5796296e-03 5.6130579e-03 3.8428861e-04 6.5006791e-03]
#  [7.8008338e-12 9.9999559e-01 5.8024283e-09 1.3334631e-09 7.3742584e-10
#   2.8086606e-09 1.5597324e-08 3.2867920e-06 1.0482211e-06 3.6903641e-10]
#  [4.1876407e-03 3.5607174e-02 3.6326755e-02 8.7942067e-04 8.2082933e-01
#   4.3756249e-03 3.2814741e-02 2.2978229e-02 1.2434252e-02 2.9566865e-02]
#  [5.6004566e-05 6.7087392e-05 1.5473304e-05 2.0297694e-03 1.6155109e-02
#   7.9367537e-04 1.6898764e-07 1.0910908e-04 1.1080807e-04 9.8066276e-01]
#  [5.7060813e-04 3.2189849e-04 1.2708381e-04 3.3121853e-04 6.5650546e-04
#   6.3688946e-01 2.9432920e-01 7.9090460e-05 6.6184528e-02 5.1037350e-04]
#  [4.8735156e-08 1.5528258e-07 1.5692036e-08 1.7270091e-05 5.7649104e-05
#   1.2968636e-06 6.6295415e-12 1.0022595e-05 2.0141210e-06 9.9991155e-01]]
# [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
# [7 2 1 0 4 1 4 9 5 9]