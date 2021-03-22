import numpy as np

#1. 데이터
x_train = np.load('../data/npy/cifar100_x_train.npy')
y_train = np.load('../data/npy/cifar100_y_train.npy')
x_test = np.load('../data/npy/cifar100_x_test.npy')
y_test = np.load('../data/npy/cifar100_y_test.npy')

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])

print('y_train[0]', y_train[0])
print(x_train[0].shape)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66)

#1.1 데이터 전처리
print(x_train.shape) 
print(x_test.shape) 
print(x_val.shape) 
# (40000, 32, 32, 3)
# (10000, 32, 32, 3)
# (10000, 32, 32, 3)
x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

#tensorflow _to_categorical
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
# (40000, 100)
# (10000, 100)
# (10000, 100)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,
                padding = 'SAME', input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])))
            
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(120, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(50, kernel_size=(3,3), padding = 'SAME'))
model.add(Dropout(0.2))
model.add(Conv2D(30, kernel_size=(3,3), padding = 'SAME'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(30, kernel_size=(3,3), padding = 'SAME'))
model.add(Conv2D(30, kernel_size=(3,3), padding = 'SAME'))
model.add(Flatten())
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'softmax'))
model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
modelpath = '../data/modelCheckpoint/k46_cifar10_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 64, validation_data= (x_val, y_val), verbose = 1 ,callbacks = [early_stopping, cp])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

print('loss : ', loss)
print('y_pred :', np.argmax(y_test[:5], axis=-1))
print('y_test :', np.argmax(y_test[:5], axis=-1))

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

# loss :  [2.688469171524048, 0.33550000190734863]
# y_pred : [95 97 72 21 23]
# y_test : [49 33 72 51 71]