#다 : 다 mlp

# 실습
# 1. R2      : 0.5이하
# 2. layer   : 5개 이상
# 3. node : 각 10개 이상
# 4. batch_size : 8이하
# 5. epoch : 30이상

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(301, 401)])
y = np.array([range(711, 811), range(1,101)])
print(x.shape) # (5, 100)
print(y.shape) # (2, 100)
x_pred2 = np.array([100,402,101,100,401])
print("x_pred2.shape :", x_pred2.shape) # (5, ) = 1차원

x = np.transpose(x)
y = np.transpose(y)
#x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5) # = 2차원으로 변경
print(x.shape)   # (100, 5)
print(y.shape)   # (100, 2)
print("x_pred2.shape : ", x_pred2.shape) # (1, 5) = 2차원으로 변환

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
     #2차원, 3차원 상관없이 행 기준으로 자름

print(x_train.shape) # (80, 5)
print(y_train.shape) # (80, 2)
print(x_pred2.shape)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print(x_train)
print(x_train.shape) #(80,5)
print(x_test.shape)  #(20,5)
print(y_train.shape) #(80,2)
print(y_test.shape)  #(20,2)

model = Sequential()
model.add(Dense(10, input_dim=5)) #데이터에 맞춰서 고정
model.add(Dense(13))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(2)) #의 피처=열=컬럼 : 데이터에 맞춰서 고정

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=40, batch_size=1, validation_split=0.2) #batch_size=1은 1개씩

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test) #계산 중 평가할 값
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