#실습 다:1 앙상블을 구현하시오

import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y1 = np.array([range(711, 811), range(1,101), range(201,301)])
#y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
#y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, shuffle = False, train_size = 0.8)


#=======================================================
#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape = (3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(5, activation = 'relu')(dense1)
output1 = Dense(3)(dense1)

# 모델2
input2 = Input(shape = (3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(middle1)
middle1 = Dense(30)(middle1)

#모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 모델 선언
model = Model(inputs = [input1, input2],
            outputs = output1) #input 2개 이상은 list 로 묶어줌
model.summary()

#=======================================================


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=1, validation_split=0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)

print(loss)
print("model.metrics_names : ", model.metrics_names)

y1_predict = model.predict([x1_test, x2_test])

print("=======================")
print("y_predict : \n", y1_predict)
print("=======================")

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
RMSE1 = RMSE(y1_test, y1_predict)
print("RMSE :" , RMSE1)

#R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)
R2 = R2(y1_test, y1_predict)
print("R2(y1_test) : ", R2)

# #예측값 추출
# y_pred2 = model.predict(x_pred2)
# print("y_pred2 : ", y_pred2)