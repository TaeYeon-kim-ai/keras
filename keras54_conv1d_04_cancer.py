#사이킷런 데이터셋
#LSTM으로 모델링
#Dense와 성능비교
#이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names) # 데이터셋 컬럼명 확인

x = datasets.data
y = datasets.target
print(x.shape)#(569, 30)
print(y.shape)#(569,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#1.1 전처리 / minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)



#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
input1 = Input(shape=(x.shape[1] ,1))
conv1d = Conv1D(50, 2, activation='relu')(input1)
dense1 = MaxPooling1D(pool_size=1)(conv1d)
conv1d = Conv1D(50, 2, activation='relu')(dense1)
dense1 = Dense(36, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1) 
dense1 = Dense(40, activation='relu')(dense1)
dense1 = Dense(40, activation='relu')(dense1)
outputs = Dense(1, activation= 'sigmoid')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# input1 = Input(shape=(x.shape[1],1))
# lstm = LSTM(50, activation='relu')(input1)
# dense1 =  Dense(50, activation='relu')(lstm)
# dense1 = Dense(36, activation='relu')(dense1)
# dense1 = Dense(40, activation='relu')(dense1) 
# dense1 = Dense(40, activation='relu')(dense1)
# dense1 = Dense(40, activation='relu')(dense1)
# outputs = Dense(1, activation= 'sigmoid')(dense1)
# model = Model(inputs = input1, outputs = outputs)
# model.summary()

#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 30, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_predict = model.predict(x_test[:10])
# y_pred = list(map(int,np.round(y_predict,0)))
result = np.transpose(y_predict)
print(np.argmax(result[:5], axis=-1))
print(np.argmax(y_test[:5], axis=-1))

#Conv1D
# loss :  0.5216351747512817
# acc :  0.7649727463722229

#LSTM
# loss :  0.18323636054992676
# acc :  0.9561403393745422