# 45번 카피해서 57번에 복붙
#다차원 댄스 모델
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

y_train = x_train #출력값을 유지하기 위한 test ver
y_test = x_test
print(y_train.shape)
print(x_test.shape)
# (60000, 28, 28, 1)
# (10000, 28, 28, 1)

#2. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Reshape
#4차원 shape도  Dense형태로 가능하나, 출력할 때 Flatten을 해야한다.
model = Sequential()
model.add(Dense(64, input_shape = (28, 28, 1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(784, activation='relu'))  # 아래 reshape해주는 레이어와 [0], [1], [2] 이 곱했을 때 맞아야함 노드의 갯수를 맞춰야함.
model.add(Reshape((28, 28, 1)))#reshape시켜줌 : 연산레이어가 아닌 위에서 받은 레이어를 잘라주는 레이어
                               #디폴트 값이 있기 때문에 (())<-를 두개 사용 이 후 배울 예정
model.add(Dense(1)) #1로 출력
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor = 'loss', patience = 6, mode = 'auto')
modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'     
# k45_mnist_37_0100(0.0100).hdf5
cp = ModelCheckpoint(filepath= modelpath , monitor='val_loss', save_best_only=True, mode = 'auto')
#filepath='(경로)' : 가중치를 세이브 해주는 루트
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split=0.5, verbose = 1 ,callbacks = [early_stopping, cp, reduce_lr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=16)
print('loss : ', loss)
y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)

# loss :  [0.0957411602139473, 0.975600004196167]
# [7 2 1 0 4 1 4 9 5 9]
 