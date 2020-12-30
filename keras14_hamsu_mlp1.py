#다 : 1 mlp 함수형
#keras10_mlp2.py를 함수로 바꾸시오
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))
print(x.shape) # (3, 100)
print(y.shape) # (100, )
x = np.transpose(x)
print(x)
print(x.shape)   # (100, 3) #단지 열행 위치 바꿔줌

x_pred2 = np.array([100,402,101])
print("x_pred2.shape :", x_pred2.shape) # (3, ) = 1차원
#x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 3) # = 2차원으로 변경
print(x.shape)   # (100, 3)
print(y.shape)   # (100, 1)
print("x_pred2.shape : ", x_pred2.shape) # (1, 5) = 2차원으로 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80,)

#2. 모델구성
input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1) 
dense3 = Dense(4)(dense2)
outputs = Dense(1)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test,y_test) 
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) 
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