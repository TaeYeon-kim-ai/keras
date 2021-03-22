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
#4차원 shape도  Dense형태로 가능하나, 출력할 때 Flatten을 해야한다.
model = Sequential()
model.add(Dense(64, input_shape = (28, 28, 1)))
model.add(Dense(64))
model.add(Flatten())
# model.add(Conv2D(150, kernel_size=(3,3), padding = 'SAME'))
# model.add(Conv2D(1, kernel_size=(3,3), padding = 'SAME'))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor = 'loss', patience = 6, mode = 'auto')
modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'     
# k45_mnist_37_0100(0.0100).hdf5
cp = ModelCheckpoint(filepath= modelpath , monitor='val_loss', save_best_only=True, mode = 'auto')
#filepath='(경로)' : 가중치를 세이브 해주는 루트
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split=0.5, verbose = 1 ,callbacks = [early_stopping, cp, reduce_lr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
print('loss : ', loss)

y_pred = model.predict(x_test[:10])
print(y_pred)
print(y_test[:10])
print(np.argmax(y_test[:10], axis=-1))

# 시각화
import  matplotlib.pyplot as plt


plt.figure(fihsize = (10,6))

plt.subplot(2,1,1) #2행 1열 중 첫번때
plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('cost_loss') #loss,cost #타이틀깨진것 한글처리 해둘 것
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2) #2행 2열중 2번째
plt.plot(hist.history['acc'], marker = '.', c='red')
plt.plot(hist.history['val_acc'], marker = '.', c='blue')
plt.grid() #그래프 격자(모눈종이 형태)

plt.title('cost_acc')  #타이틀깨진것 한글처리 해둘 것
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# loss :  [0.0957411602139473, 0.975600004196167]
# [7 2 1 0 4 1 4 9 5 9]

