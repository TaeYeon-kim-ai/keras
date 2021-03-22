#Dropout적용

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names) # 데이터셋 컬럼명 확인

x = datasets.data
y = datasets.target
print(x.shape) #(569, 30)
print(y.shape) #(569,)
print(x[:5])
print(y)

#1.1 전처리 / minmax, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)



#2. 모델링
#각 노드의 Dropout, 노드수, activation을 여러 경우의 수를 활용하여 튜닝가능
# a = [0.1, 0.2, 0.3]
# b = [0.1, 0.2, 0.3]
# c = [100, 200, 300]
# d = ['relu', 'linear', 'elu', 'selu', 'tanh']
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
input1 = Input(shape = (30,))
dense1 = Dense(1200, activation='sigmoid')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(80, activation='sigmoid')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(70, activation='sigmoid')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(100, activation='sigmoid')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(50, activation='sigmoid')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(50, activation='sigmoid')(dense1)
dense1 = Dropout(0.1)(dense1)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=(30,)))
# model.add(Dense(1, activation='sigmoid'))


#3. 컴파일,
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'acc', patience = 30, mode = 'auto')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test[:10])
y_predict = model.predict(x_test[:10])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_predict)
print(y_pred)
print(y_test[:10])
#실습1. acc 0.985이상 올릴 것
#실습2. predict 출력해볼것
#y[-5:-1] = ??

# loss :  0.5385931134223938
# acc :  0.9736841917037964

# loss :  0.05537671223282814
# acc :  0.9824561476707458

# loss :  0.05472574755549431
# acc :  0.9912280440330505
# [[0.98721755 0.98863095 0.9889385  0.9889311  0.9870265  0.00612339
#   0.00251615 0.98906994 0.9861438  0.9887173 ]]
# [1 1 1 1 1 0 0 1 1 1]
# [1 1 1 1 1 0 0 1 1 1]

#Dropout
# loss :  0.0540630929172039
# acc :  0.9824561476707458
# [[0.98637134 0.9897046  0.9906318  0.9905861  0.98537487 0.00800762
#   0.0036923  0.99085176 0.9829979  0.9901099 ]]
# [1 1 1 1 1 0 0 1 1 1]
# [1 1 1 1 1 0 0 1 1 1]

'''
Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        : 속성 정보 :
        -반경 (중심에서 주변 지점까지의 거리 평균)
        -텍스처 (회색조 값의 표준 편차)
        -둘레
        - 지역
        -부드러움 (반경 길이의 국부적 변동)
        -콤팩트 함 (둘레 ^ 2 / 면적-1.0)
        -오목 함 (윤곽의 오목한 부분의 심각도)
        -오목한 점 (윤곽의 오목한 부분의 수)
        -대칭
        -프랙탈 차원 ( "해안선 근사치"-1)
'''