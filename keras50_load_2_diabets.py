#실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지 총 6개의 파일을 완성하시오.

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x_data = np.load('../data/npy/diabets_x.npy')
y_data = np.load('../data/npy/diabets_y.npy')

#1. 데이터 
x = x_data
y = y_data

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)
print(np.max(x), np.min(x))
#print(dataset.DESCR)

#x = x / np.max(x)
#print(np.max(x), np.min(x)) # 정규화

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size= 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
model.add(Dense(75, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
modelpath = '../data/modelCheckpoint/k46_diabets_{epoch:02d}-{val_loss:.4f}.hdf5)'
cp = ModelCheckpoint(filepath= modelpath, monitor= 'val_loss', save_best_only=True, mode = 'auto')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping, cp])

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

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label = 'val_loss')
plt.grid()

plt.title('cost_loss') #loss, cost
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')


plt.subplot(2,1,2)
plt.plot(hist.history['mae'], marker = '.', c = 'red', label = 'mae')
plt.plot(hist.history['val_mae'], marker = '.', c = 'blue', label = 'val_mae')
plt.grid()

plt.title('cost_mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.show()

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

#Earlystopping 적용
# loss :  57.708213806152344
# mae :  5.144794464111328
# RMSE :  7.596591897809446
# R2 :  0.991656001135139 ---??? 뭔가잘못된

# loss :  4958.51513671875
# mae :  55.56264114379883
# RMSE :  70.41672085545045
# R2 :  0.2830510408034974

#model = Sequential()
# model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
# model.add(Dense(75, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(1))