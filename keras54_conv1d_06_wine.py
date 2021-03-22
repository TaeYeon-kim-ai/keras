import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(x.shape[0], x.shape[1], 1)

#  keras.to_categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

print(x_train.shape) # (113, 13, 1)
print(x_test.shape) #(36, 13, 1)
print(x_val.shape) #(29, 13, 1)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Input, LSTM
# input1 = Input(shape=(x.shape[1] ,1))
# conv1d = Conv1D(100, 2, padding='SAME')(input1)
# maxp = MaxPooling1D(2)(conv1d)
# conv1d = Conv1D(100, 2, activation='relu', padding='SAME')(maxp)
# drop = Dropout(0.2)(conv1d)
# fla = Flatten()(drop)
# dense1 = Dense(36, activation='relu')(fla)
# dense1 = Dense(40, activation='relu')(dense1) 
# dense1 = Dense(40, activation='relu')(dense1)
# dense1 = Dense(40, activation='relu')(dense1)
# outputs = Dense(3, activation= 'softmax')(dense1)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

input1 = Input(shape=(x.shape[1],1))
lstm = LSTM(100, activation='relu')(input1)
dense1 =  Dense(100, activation='relu')(lstm)
drop = Dropout(0.2)(dense1)
dense1 = Dense(36, activation='relu')(drop)
dense1 = Dense(40, activation='relu')(dense1) 
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
outputs = Dense(3, activation= 'softmax')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()


#3. 컴파일, 훈련
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
print(y_train[-5:-1])

#y값 중에서 가장 큰 값을 1로 바꾼다 : argmax
#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = -1))
#print(np.argmax(y_pred, axis = 2)) 

#Conv1d
# loss :  0.00042515600216574967
# acc :  1.0

#LSTM
# loss :  0.3086625039577484
# acc :  0.9166666865348816