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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (28, 28, 1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
model.summary()

#model.save('../data/h5/k52_1_model1.h5')#모델저장

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
#modelpath = '../data/modelCheckpoint/k52_1_MCK.h5_{epoch:02d}-{val_loss:.4f}.hdf5'  
# k52_1_mnist_??? => k52_1_MCK.h5이름 바꿔줄 것   
# k45_mnist_37_0100(0.0100).hdf5
#cp = ModelCheckpoint(filepath= modelpath , monitor='val_loss', save_best_only=True, mode = 'auto')
# filepath='(경로)' : 가중치를 세이브 해주는 루트
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
#model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split=0.2, verbose = 1 ,callbacks = [early_stopping, cp])

# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')


#모델 save -- load
# model = load_model('../data/h5/k52_1_model2.h5')
#4. 평가, 예측
# loss1 = model.evaluate(x_test, y_test, batch_size=8)
# print('loss : ', loss1)
# y_pred1 = model.predict(x_test[:10])
# print(y_pred1)
# print(y_test[:10])
# print(np.argmax(y_test[:10], axis=-1))
# loss :  [0.317549467086792, 0.886900007724762] #튜닝 다시


#weight save -- load
#weight는 결과만 불러오므로 모델과 컴파일은 필요함 midel.fit도
model.load_weights('../data/h5/k52_1_weight.h5')
result = model.evaluate(x_test, y_test, batch_size=8)
print('weight_loss : ', result[0])
print('weight_acc : ', result[1])
#print(np.argmax(y_test[:10], axis=-1))
# loss :  [0.317549467086792, 0.886900007724762] #튜닝 다시

# 기존
# loss :  [0.317549467086792, 0.886900007724762] #튜닝 다시
# [9 2 1 1 6 1 4 6 5 7]

model2 = load_model('../data/h5/k52_1_model2.h5')
result2 = model.evaluate(x_test, y_test, batch_size=8)
print('weight_loss : ', result2[0])
print('weight_acc : ', result2[1])
#result2 = model.predict(x_test[:10])
#print(np.argmax(y_test[:10], axis=-1))