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

x_train = x_train.reshape(-1,32*32,3).astype('float32')/255
x_test= x_test.reshape(-1,32*32,3).astype('float32')/255

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = one.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = one.transform(y_test.reshape(-1,1)).toarray()

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Input, LSTM
input1 = Input(shape=(32*32, 3))
conv1d = Conv1D(100, 2, padding='SAME')(input1)
maxp = MaxPooling1D(2)(conv1d)
conv1d = Conv1D(100, 2, activation='relu', padding='SAME')(maxp)
drop = Dropout(0.2)(conv1d)
fla = Flatten()(drop)
dense1 = Dense(36, activation='relu')(fla)
dense1 = Dense(40, activation='relu')(dense1) 
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
outputs = Dense(100, activation= 'softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# input1 = Input(shape=(32*32, 3))
# lstm = LSTM(100, activation='relu')(input1)
# dense1 = Dense(100, activation='relu')(lstm)
# drop = Dropout(0.2)(dense1)
# dense1 = Dense(36, activation='relu')(drop)
# dense1 = Dense(40, activation='relu')(dense1) 
# dense1 = Dense(40, activation='relu')(dense1)
# dense1 = Dense(40, activation='relu')(dense1)
# outputs = Dense(100, activation= 'softmax')(dense1)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_split= 0.2, verbose = 1 ,callbacks = [early_stopping])

#4. 평가, 예측
loss , acc= model.evaluate(x_test, y_test)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

#Conv1D
#loss :  [3.3595569133758545, 0.23090000450611115]

#LSTM
#loss :  [nan, 0.10000000149011612]