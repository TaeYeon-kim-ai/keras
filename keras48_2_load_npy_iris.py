import numpy as np
#데이터 불러오기
x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape)
print(y_data.shape)

x = x_data
y = y_data
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(x.shape) #(150, 4)
# print(x[:5])

#1.1 전처리 / minmax, train_test_split

#원핫인코딩 OneHotEncofing
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
#y = to_categorical(y)
# y_train = to_categorical(y_train) #to_categorical 적용
# y_test = to_categorical(y_test) #to_categorical 적용\

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                          #. Set
y = one.transform(y).toarray()      #. transform
# print(y)
print(x.shape) #(150,4)
print(y.shape) #(150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
input1 = Input(shape = (x_train.shape[1],)) 
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(400, activation='relu')(dense1)
dense1 = Dense(250, activation='relu')(dense1)
dense1 = Dense(100, activation='relu')(dense1)
dense1 = Dense(20, activation='relu')(dense1)
outputs = Dense(3, activation='softmax')(dense1)#원핫인코더한 수와 동일
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k46_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_train[-5:-1])

#y값 중에서 가장 큰 값을 1로 바꾼다 : argmax
#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = -1))
#print(np.argmax(y_pred, axis = 2)) 

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

plt.title('cost_acc')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

# loss :  0.09326354414224625
# acc :  0.9666666388511658
# [[4.4882458e-06 1.1504701e-03 9.9884498e-01]
#  [1.0000000e+00 1.2430774e-08 5.5269052e-17]
#  [9.9465996e-01 5.3400816e-03 2.2348919e-08]
#  [8.3166477e-04 2.2638336e-01 7.7278501e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
# [2 0 0 2]