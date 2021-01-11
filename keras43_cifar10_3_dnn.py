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

#tensorflow _to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

print(y_train.shape) 
print(x_train.shape)
# (40000, 10)
# (40000, 3072)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
input1 = Input(shape=(16*48*4,))
dense1 = Dense(200, activation='relu', input_shape = (16*48*4,))(input1)
dense1 = Dense(100, activation= 'relu')(dense1)
dense1 = Dense(80, activation= 'relu')(dense1)
dense1 = Dense(60, activation= 'relu')(dense1)
dense1 = Dense(40, activation= 'relu')(dense1)
dense1 = Dense(30, activation= 'relu')(dense1)
dense1 = Dense(20, activation= 'relu')(dense1)
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
# [[0.01401374 0.0270426  0.06957036 0.11014512 0.10192871 0.07001724
#   0.53466463 0.03934649 0.00797646 0.02529465]
#  [0.0139803  0.02692783 0.06944458 0.10957904 0.10209226 0.06970914
#   0.5358754  0.03929657 0.00790744 0.02518739]
#  [0.01389117 0.02652587 0.07017696 0.10883962 0.10384309 0.07042421
#   0.53391176 0.03960177 0.00796256 0.02482297]
#  [0.01382817 0.02628834 0.07020663 0.1079722  0.10449085 0.07022642
#   0.53487414 0.03961701 0.00789245 0.0246037 ]
#  [0.01382659 0.02631156 0.06988195 0.1077214  0.10410505 0.06975498
#   0.53647363 0.03948393 0.00781888 0.02462211]
#  [0.0139168  0.02666535 0.06969762 0.10886057 0.1030084  0.0698321
#   0.53578115 0.03940387 0.0078868  0.02494731]
#  [0.0139275  0.02667578 0.07002374 0.10924479 0.10330531 0.0703422
#   0.5340109  0.03953715 0.00797269 0.02496007]
#  [0.01388434 0.02653833 0.06975687 0.10844477 0.10339294 0.06979422
#   0.5360675  0.03942982 0.00786073 0.0248305 ]
#  [0.01375559 0.02598821 0.0705625  0.10720429 0.10564074 0.0704608
#   0.534409   0.03976596 0.00788335 0.02432956]
#  [0.01404894 0.02720754 0.06923652 0.11040595 0.10118954 0.06966981
#   0.53564626 0.03920723 0.00794441 0.02544375]]
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# [3 8 8 0 6 6 1 6 3 1]