#Dropout적용

import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
# x, y = load_iris(return_X_y=True)

dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.DESCR)
print(dataset.feature_names)
print(x.shape) #(150, 4)
print(x[:5])

#1.1 전처리 / minmax, train_test_split

#원핫인코딩 OneHotEncofing
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
#y = to_categorical(y)
# y_train = to_categorical(y_train) #to_categorical 적용
# y_test = to_categorical(y_test) #to_categorical 적용\

#  sklearn.onehotencoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()           
y = y.reshape(-1,1)                 #. y_train => 2D
one.fit(y)                          #. Set
y = one.transform(y).toarray()      #. transform
print(y)
print(x.shape) #(150,4)
print(y.shape) #(150,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
input1 = Input(shape = (4,)) 
dense1 = Dense(500, activation='relu')(input1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(60, activation='relu')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(120, activation='relu')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(90, activation='relu')(dense1)
dense1 = Dense(70, activation='relu')(dense1)
outputs = Dense(3, activation='softmax')(dense1)#원핫인코더한 수와 동일
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=(30,)))
# model.add(Dense(1, activation='sigmoid'))


#3. 컴파일,
#다중분류일 경우 : 
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto') # #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 7, validation_data = (x_val, y_val), verbose = 1 ,callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[-5:-1])
print(y_pred)
print(y_train[-5:-1])

#y값 중에서 가장 큰 값을 1로 바꾼다 : argmax
#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = -1))
#print(np.argmax(y_pred, axis = 2)) 



# loss :  0.14753571152687073
# acc :  0.9666666388511658
# [[2.1295748e-11 1.3989408e-07 9.9999988e-01]
#  [1.1455371e-19 7.7679168e-17 1.0000000e+00]
#  [3.5330005e-12 1.4296412e-08 1.0000000e+00]
#  [1.3562451e-10 5.9833246e-07 9.9999940e-01]]
# [[0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
# [2 2 2 2]

# Dropout적용
# loss :  0.02586069330573082
# acc :  1.0
# [[1.6517568e-05 2.1786096e-02 9.7819740e-01]
#  [1.1007625e-08 9.8477839e-04 9.9901521e-01]
#  [1.1485755e-05 1.8228820e-02 9.8175967e-01]
#  [2.2588742e-05 2.8938046e-02 9.7103935e-01]]
# [[0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
# [2 2 2 2]

'''
** 데이터 세트 특성 : **

    : 인스턴스 수 : 150 개 (3 개 클래스 각각 50 개)
    : 속성 수 : 4 개의 숫자, 예측 속성 및 클래스
    : 속성 정보 :
        -꽃받침 길이 (cm)
        -꽃받침 너비 (cm)
        -꽃잎 길이 (cm)
        -꽃잎 너비 (cm)
        - 수업:
                -아이리스-세토 사
                -Iris-Versicolour
                -Iris-Virginica

    : 요약 통계 :

    ============== ==== ==== ======= ===== ================ ====
                    최소 최대 평균 SD 클래스 상관 관계
    ============== ==== ==== ======= ===== ================ ====
    꽃받침 길이 : 4.3 7.9 5.84 0.83 0.7826
    꽃받침 너비 : 2.0 4.4 3.05 0.43 -0.4194
    꽃잎 길이 : 1.0 6.9 3.76 1.76 0.9490 (높음!)
    꽃잎 폭 : 0.1 2.5 1.20 0.76 0.9565 (높음!)
    ============== ==== ==== ======= ===== ================ ====

    : 속성 값 누락 : 없음
    : 계급 분포 : 3 개 계급 각각 33.3 %.
    : 크리에이터 : R.A. 어부
    : 기부 : Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    : 날짜 : 1988 년 7 월
'''
