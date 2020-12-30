#다 : 다 mlp 함수형
#keras10_mlp3.py를 함수형으로 바꾸시오

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1,101), range(201,301)])
print(x.shape) # (3, 100)
print(y.shape) # (3, 100)
x = np.transpose(x)
y = np.transpose(y)

x_pred2 = np.array([100,402,101])
print("x_pred2.shape :", x_pred2.shape) # (3, ) = 1차원
#x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 3) # = 2차원으로 변경
print(x.shape)   # (100, 3)
print(y.shape)   # (100, 3)
print("x_pred2.shape : ", x_pred2.shape) # (1, 5) = 2차원으로 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
     #2차원, 3차원 상관없이 행 기준으로 자름

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(3,)) #레이어 구성
dense1 = Dense(5, activation='relu')(input1) #삽입
dense2 = Dense(3)(dense1) 
dense3 = Dense(4)(dense2)
outputs = Dense(3)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) #batch_size=1은 1개씩

#4. 평가, 예측
loss, mae = model.evaluate(x_test,y_test) #계산 중 평가할 값
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) # 예측값 RMSE에 y_test값은(20이라) x_test값을 맞춰줘야함(20,)로
#print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )

#예측값 추출
y_pred2 = model.predict(x_pred2)
print("y_pred2 : ", y_pred2)




