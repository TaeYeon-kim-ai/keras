#keras21_cancer1.py를 다중분류로 코딩하시오

import numpy as np
x_data = np.load('../data/npy/diabets_x.npy')
y_data = np.load('../data/npy/diabets_y.npy')

#1. 데이터
print(x_data.DESCR)
print(x_data.feature_names) # 데이터셋 컬럼명 확인

x = x_data
y = y_data
print(x.shape) #(569, 30)
print(y.shape) #(569,)
print(x[:5])
print(y)

#1.1 전처리 / minmax, train_test_split

#원핫코딩 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (30,))
dense1 = Dense(20, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(20, activation='relu')(dense4)
outputs = Dense(2, activation='sigmoid')(dense5)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=(30,)))
# model.add(Dense(1, activation='sigmoid'))


#3. 컴파일,
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
modelpath = '../data/modelCheckpoint/k46_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath= modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 7 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping, cp])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test[-5:-1])
print(y_train[-5:-1])
print(y_pred)

#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = 1))


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('cost_loss') #loss, cost
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()

plt.title('cost_mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()


#실습1. acc 0.985이상 올릴 것
#실습2. predict 출력해볼것
#y[-5:-1] = ??

# loss :  0.15305033326148987
# acc :  0.9824561476707458
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [1. 0.]]
# [[5.0966210e-12 9.9989939e-01]
#  [8.5732776e-01 1.3080690e-09]
#  [6.4385594e-11 9.9974132e-01]
#  [4.7192850e-10 9.9754477e-01]]
# [1 0 1 1]