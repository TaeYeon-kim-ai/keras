import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names) # 데이터셋 컬럼명 확인

x = datasets.data
y = datasets.target
# print(x.shape) #(569, 30)
# print(y.shape) #(569,)
# print(x[:5])
# print(y)

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
y=y.reshape(-1, 1)
yoh=OneHotEncoder()
yoh.fit(y)
y=yoh.transform(y).toarray()#데미변수 만들기?

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x = x.reshape(x.shape[0], x.shape[1], 1, 1) #(569, 30, 1, 1)

print(y.shape) #(569, 2)
print(x.shape) #(569, 30, 1, 1)

#1.1 전처리 / minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (x.shape[0], x.shape[1], 1, 1))) 
model.add(Conv2D(500, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(200, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(2, activation= 'sigmoid'))

#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 30, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 64 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test[:10])
y_predict = model.predict(x_test[:10])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_predict)
print(y_pred)
print(y_test[:10])
#실습1. acc 0.985이상 올릴 것
#실습2. predict 출력해볼것
#y[-5:-1] = ??

# loss :  0.5385931134223938
# acc :  0.9736841917037964

# loss :  0.05537671223282814
# acc :  0.9824561476707458

# loss :  0.05472574755549431
# acc :  0.9912280440330505
# [[0.98721755 0.98863095 0.9889385  0.9889311  0.9870265  0.00612339
#   0.00251615 0.98906994 0.9861438  0.9887173 ]]
# [1 1 1 1 1 0 0 1 1 1]
# [1 1 1 1 1 0 0 1 1 1]
