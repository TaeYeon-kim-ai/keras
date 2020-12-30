#다 : 1 mlp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense
'''
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) # (10,)

x = np.array([[1,2,3,4,5,6,7,8,9,10]
            ,[1,2,3,4,5,6,7,8,9,10]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) # (2,10)

# 행열 구분
#가장자리 [] 날리고, 작은단위(벡터) 부터 셀 것
#1. [[1,2,3], [4,5,6]]   --------------- (2, 3)
#2. [[1,2],[3,4], [5,6]] --------------- (3, 2)
#3. [[[1,2,3], [4,5,6]]] --------------- (1, 2, 3)
#4. [[1,2,3,4,5,6]] -------------------- (1, 6)
#5. [[[1,2],[3,4]], [[5,6], [7,8]]] ---- (2, 2, 2)
#6. [[1],[2],[3]] ---------------------- (3, 1)
#7. [[[1],[2]], [[3],[4]]] ------------- (2, 2, 1)
'''
'''
x = np.array([[1,2,3,4,5,6,7,8,9,10]
            ,[11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)

print(x.reshape(10,-1))

print(x.)
'''
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10]
            ,[11,12,13,14,15,16,17,18,19,20]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.transpose(x)

print(x)

print(x.shape)   # (10, 2)

#2. 모델구성
model = Sequential()
model.add(Dense(10), input_dim=2)
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2)
 

#4. 평가, 예측
loss, mae = model.evaluate(x,  y)
print('loss : ', loss)
print('mae : ', mae)



'''
#사이킷런
#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )
'''



























