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

print(x_train.shape) # (48000, 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)
print(x_val.shape) # (12000, 28, 28, 1)

#tensorflow _to_categorica
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LSTM, Input
input1 = Input(shape = (14*14, 4))
lstm = LSTM(400, activation='relu', input_shape = (14*14, 4))(input1)
dense1 = Dense(300, activation='relu')(lstm)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
outputs = Dense(10, activation='softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# loss :  [0.34222573041915894, 0.8781999945640564] - LSTM
# [[9.55988544e-13 7.09974123e-16 4.76738701e-12 3.57960359e-15
#   1.15044564e-10 8.77122366e-05 1.07733293e-11 7.18230614e-04
#   2.71819474e-08 9.99194086e-01]
#  [2.30827482e-10 1.28232382e-19 9.99526620e-01 3.28054771e-12
#   4.71825362e-04 6.20916394e-23 1.58293437e-06 7.19612024e-28
#   6.48286195e-17 7.87451684e-24]
#  [1.13207600e-13 1.00000000e+00 8.40737974e-20 2.09009414e-08
#   2.80862878e-14 3.12465093e-26 3.08461806e-13 8.53288646e-32
#   1.56645372e-23 5.31125679e-34]
#  [4.50636438e-11 9.99999285e-01 1.08781456e-15 7.00752651e-07
#   7.46336846e-12 2.97641007e-20 6.75944994e-11 6.42716041e-25
#   1.04773292e-18 9.54171715e-27]
#  [1.44702479e-01 4.10217646e-04 1.39371082e-02 9.05089639e-03
#   3.75763280e-03 1.33081548e-05 8.26935947e-01 3.36047356e-06
#   1.18788506e-03 1.10699204e-06]
#  [1.19152886e-12 9.99999881e-01 3.44618849e-18 7.91798911e-08
#   1.72028339e-13 1.52102985e-23 1.96484365e-12 9.38561574e-29
#   1.45260226e-21 6.62312497e-31]
#  [1.23355392e-07 4.60964088e-07 1.58408675e-02 4.27996781e-07
#   9.78753209e-01 9.99744731e-10 5.40451333e-03 3.14764152e-11
#   3.28388467e-07 3.86569748e-10]
#  [8.14029306e-07 2.24939729e-08 9.13432799e-04 5.39886464e-08
#   2.88745333e-02 1.75315728e-14 9.70210969e-01 7.54795640e-13
#   9.32742310e-08 9.73292492e-14]
#  [1.08847578e-21 4.50336979e-29 3.09692377e-22 4.51051759e-31
#   4.21568498e-30 9.99999642e-01 4.07092641e-20 4.01896926e-07
#   1.66266983e-11 8.07678091e-10]
#  [1.23007880e-24 1.54300214e-21 2.13953476e-21 4.96857213e-22
#   1.68873442e-26 4.96135384e-04 9.22208501e-23 9.99502182e-01
#   4.92700269e-10 1.70040312e-06]]
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# [9 2 1 1 6 1 4 6 5 7]