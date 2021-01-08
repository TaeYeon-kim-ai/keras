#Dropout적용

import numpy as np
from sklearn.datasets import load_diabetes

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
#print(dataset.DESCR)

#x = x / np.max(x)
#print(np.max(x), np.min(x)) # 정규화

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size= 0.8, shuffle = True, random_state = 66, )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
model.add(Dropout(0.1))
model.add(Dense(75, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
**Data Set Characteristics:**
  :Number of Instances: 442
  :Number of Attributes: First 10 columns are numeric predictive values
  :Target: Column 11 is a quantitative measure of disease progression one year after baseline
  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, T-Cells (a type of white blood cells)
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, thyroid stimulating hormone
      - s5      ltg, lamotrigine
      - s6      glu, blood sugar level
"""

#데이터 전처리 전
# loss :  3317.64599609375
# mae :  47.06387710571289
# RMSE :  57.59901189718801
# R2 :  0.488809627121195

#데이터 엉망 처리 후
# loss :  3379.458984375
# mae :  47.35618591308594
# RMSE :  58.13311275393621
# R2 :  0.47928539874511966

#데이터 x를 전처리한 후
# loss :  3291.452880859375
# mae :  46.496116638183594
# RMSE :  57.37118551454562
# R2 :  0.49284554101046385

#데이터 x_train잡아서 전처리한 후....
# loss :  3421.5537109375
# mae :  47.82155227661133
# RMSE :  58.49405010020266
# R2 :  0.47279929140593635

#발리데이션 test분리
# loss :  3369.262451171875
# mae :  48.33604431152344
# RMSE :  58.04534944194592
# R2 :  0.5128401315682825

#Dropout적용  // earlystopping 48
# loss :  3335.88720703125
# mae :  47.15842819213867
# RMSE :  57.75713979884684
# R2 :  0.5176658618735176

#Earlystopping 적용
# loss :  57.708213806152344
# mae :  5.144794464111328
# RMSE :  7.596591897809446
# R2 :  0.991656001135139 ---??? 뭔가잘못된

#model = Sequential()
# model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
# model.add(Dense(75, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(1))