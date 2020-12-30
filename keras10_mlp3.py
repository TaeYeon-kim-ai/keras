#다 : 다 mlp
#내일의 데이터 예측(변수마다의)
# y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (?)

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1,101), range(201,301)])
print(x.shape) # (3, 100)
print(y.shape) # (3, 100)
x = np.transpose(x)
y = np.transpose(y)
print(x)
print(y)
print(x.shape)   # (100, 3)
print(y.shape)   # (100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
     #2차원, 3차원 상관없이 행 기준으로 자름

print(x_train.shape) # (80, 3)
print(y_train.shape) # (80, 3)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(x_train)
print(x_train.shape) #(80,3)
print(x_test.shape)  #(20,3)
print(y_train.shape) #(80,3)
print(y_test.shape)  #(20,3)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))

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






