# 다: 다+1 앙상블을 구현하시오
import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])

y1 = np.array([range(711, 811), range(1,101), range(201,301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])
y3 = np.array([range(601, 701), range(811, 911), range(1100, 1200)])

x_pred2 = np.array([602,812,1101])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)
x_pred2 = x_pred2.reshape(1, 3) # = 2차원으로 변경

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, shuffle = False, train_size = 0.8)
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, shuffle = False, train_size = 0.8)

#=======================================================
#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape = (3,))
dense1 = Dense(10, activation = 'relu')(input1)
dense1 = Dense(13, activation = 'relu')(dense1)
dense1 = Dense(5, activation = 'relu')(dense1)
dense1 = Dense(7, activation = 'relu')(dense1)
output1 = Dense(3)(dense1)

# 모델2
input2 = Input(shape = (3,))
dense2 = Dense(10, activation = 'relu')(input2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
dense2 = Dense(5, activation = 'relu')(dense2)
output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(20)(merge1)
middle1 = Dense(30)(middle1)
middle1 = Dense(30)(middle1)

#모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

#모델 분기 2
output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

#모델 분기 3
output3 = Dense(15)(middle1)
output3 = Dense(7)(output3)
output3 = Dense(7)(output3)
output3 = Dense(3)(output3)

# 모델 선언
model = Model(inputs = [input1, input2],
            outputs = [output1, output2, output3]) #input 2개 이상은 list 로 묶어줌
model.summary()
#=======================================================


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=70, batch_size=1, validation_split=0.2, verbose = 1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                    [y1_test, y2_test, y3_test], batch_size = 1)

print(loss)
#[1264.953857421875, 661.8886108398438, 603.0653686523438, 21.079633712768555, 17.948322296142578]
print("model.metrics_names : ", model.metrics_names)
#model.metrics_names :  ['loss', 'dense_13_loss', 'dense_17_loss', 'dense_13_mae', 'dense_17_mae']

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])

print("=======================")
print("y1_predict : \n", y1_predict)
print("=======================")
print("y2_predict : \n", y2_predict)
print("=======================")
print("y2_predict : \n", y3_predict)
print("=======================")
#=======================================================

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE :" , RMSE1)
print("RMSE :" , RMSE2)
print("RMSE :" , RMSE3)
print("AVG(RMSE) :" , (RMSE1 + RMSE2 + RMSE3)/3)

#R2 구하기
from sklearn.metrics import r2_score
def R2(y_test, y_predict) :
    return r2_score(y_test, y_predict)
R2_1 = R2(y1_test, y1_predict)
R2_2 = R2(y2_test, y2_predict)
R2_3 = R2(y3_test, y3_predict)
print("R2(y1_test) : ", R2_1)
print("R2(y2_test) : ", R2_2)
print("R2(y2_test) : ", R2_3)
print("AVG(R2) : ", (R2_1 + R2_2 + R2_3)/3)

# RMSE : 0.27586799140911517
# RMSE : 0.31969127419419074
# RMSE : 0.3844715459780286
# AVG(RMSE) : 0.32667693719377816
# R2(y1_test) :  0.9977111834982226
# R2(y2_test) :  0.996926240276815
# R2(y2_test) :  0.9955543347468652
# AVG(R2) :  0.9967305861739676

# #예측값 추출 ㅜㅜ 덜함
# y_pred2 = model.predict(x_pred2)
# print("y_pred2 : ", y_pred2)