#보스턴 집값
#실습:모델구성
import numpy as np
from sklearn.datasets import load_boston
#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,   )

#1_2. 데이터 전처리(MinMaxScalar)
#ex 0~711 = 최댓값으로 나눈다  0~711/711
# X - 최소값 / 최대값 - 최소값
print("===================")
print(x[:5]) # 0~4
print(y[:10]) 
print(np.max(x), np.min(x)) # max값 min값
print(dataset.feature_names)
#print(dataset.DESCR) #묘사
'''
#x = x /711
#x = (x - 최소) / (최대 - 최소)
#  = (x - np.min(x)) / (np.max(x) - np.min(x))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
'''
x = x.reshape(x.shape[0], x.shape[1], 1, 1) #(506, 13, 1, 1)
y = y.reshape(y.shape[0], 1, 1, 1) #(506, 1, 1, 1)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
x_test, x_val, y_test, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_tranin = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
# x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
# x_val = scaler.transform(x_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(filters = 100, kernel_size=(2,2), strides =1 ,padding = 'SAME', input_shape = (13, 1, 1))) 
model.add(Conv2D(500, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(200, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Conv2D(100, kernel_size=(2,2), padding = 'SAME'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
# model.add(Dropout(0.2)), 
model.add(Dense(1, activation= 'relu'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
Early_stopping = EarlyStopping(monitor = 'loss', patience=10, mode = 'auto')
model.compile(loss = 'mse', optimizer='adam', metrics = ['acc'])
model.fit(x_test, y_test, epochs=300, batch_size=32, validation_data= (x_val, y_val), verbose = 1, callbacks = [Early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_test) 
#print(y_predict)

# #RMSE 구하기
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict) : 
#     return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
# print("RMSE :" , RMSE(y_test, y_predict))

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2 )

#전처리 전
# loss :  16.55705451965332
# mae :  3.3871774673461914
# RMSE : 4.069036165639308
# R2 :  0.8019086688524137

#통째로 전처리
# loss :  11.465134620666504
# mae :  2.5706095695495605
# RMSE : 3.386020620416784
# R2 :  0.8628292327610475

#제대로 전처리(?)
# loss :  531.5300903320312
# mae :  21.24960708618164
# RMSE : 23.054936080104717
# R2 :  -5.359313211830821

#발리데이션 test분리
# loss :  5.44482421875
# mae :  1.7919334173202515
# RMSE : 2.3334145056348183
# R2 :  0.9430991642272919

#CNN적용..
# loss :  78.73698425292969
# acc :  0.0 >??

'''
#전처리가 된 데이터(정규화)
#[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00] = 되어있지않음...

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''
