# 1 : 다 앙상블

import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(711, 811), range(1,101), range(201,301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, shuffle = False, train_size = 0.8)

#=======================================================
#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape = (3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(20, activation = 'relu')(dense1)
output1 = Dense(3)(dense1)

#모델 분기 1
output1 = Dense(30)(dense1)
output1 = Dense(100)(output1)
output1 = Dense(20)(output1)
output1 = Dense(3)(output1)

#모델 분기 2
output2 = Dense(30)(dense1)
output2 = Dense(100)(output2)
output2 = Dense(20)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model=Model(inputs  =input1, outputs= [output1, output2]) #input 2개 이상은 list 로 묶어줌
model.summary()
#=======================================================


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate(x1_test,
                    [y1_test, y2_test], batch_size = 1)
print(loss)
print("model.metrics_names : ", model.metrics_names)

y1_predict, y2_predict = model.predict(x1_test)
print("=======================")
print("y1_predict : \n", y1_predict)
print("=======================")
print("y2_predict : \n", y2_predict)
print("=======================")

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE :" , RMSE1)
print("RMSE :" , RMSE2)
print("AVG(RMSE) :" , (RMSE1+RMSE2)/2)

#R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)
R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)
print("R2(y1_test) : ", R2_1)
print("R2(y2_test) : ", R2_2)
print("AVG(R2) : ", (R2_1+R2_2)/2)

# #예측값 추출
# y_pred2 = model.predict(x_pred2)
# print("y_pred2 : ", y_pred2)

