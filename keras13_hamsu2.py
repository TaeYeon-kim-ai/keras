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


#2. 모델구성 + 함수형 모델 추가
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 함수형 모델구성
'''
가와 나 모델 둘다 Seq모델이고 성능 같다
'''
#가.모델
input1 = Input(shape=(5,)) #레이어 구성
aaa = Dense(5, activation='relu')(input1) #삽입
aaa = Dense(3)(aaa) 
aaa = Dense(4)(aaa)
outputs = Dense(2)(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

'''
#나. 모델
# Model = Sequential()
# # model.add(Dense(10, input_dim=1))
# model.add(Dense(5, activation='relu', input_shape=(1,)))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose = 1) 

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test) 
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
'''