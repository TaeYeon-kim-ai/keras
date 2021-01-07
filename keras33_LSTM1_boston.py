#텐서플로 데이터셋
#LSTM으로 모델링
#DNN과 비교
import numpy as np
from tensorflow.keras.datasets import boston_housing
#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data() #이미 나누어져서 나옴(?)
print(x_train.shape) #(404, 13)
print(x_test.shape) # (102, 13)
print(y_train.shape) #(404,   )
print(y_test.shape) # (102,)

x_train = x_train.reshape(x_train.shape)

#1_2. 데이터 전처리(MinMaxScalar)
#ex 0~711 = 최댓값으로 나눈다  0~711/711
# X - 최소값 / 최대값 - 최소값
print("===================")
print(x_train[:5]) # 0~4
print(y_train[:10]) 
print(np.max(x_train), np.min(x_train)) # max값 min값
#print(dataset.feature_names)
#print(dataset.DESCR) #묘사
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import train_test_split
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size= 0.8, shuffle = True, random_state = 66, )

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(64, activation='relu')(dense3)
outputs = Dense(1)(dense4)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 300, batch_size = 11, validation_data = (x_val, y_val))

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

# dense1 = Dense(128, activation='relu')(input1)
# dense2 = Dense(128, activation='relu')(dense1) 
# dense3 = Dense(64, activation='relu')(dense2)
# dense4 = Dense(64, activation='relu')(dense3)
# loss :  13.808443069458008
# mae :  2.4139318466186523
# RMSE : 3.7159715369931208
# R2 :  0.8328379375295395