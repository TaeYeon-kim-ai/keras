#hist를 이용하여 그래프를 그리시오.
#loss, val_loss, acc, val_acc

#keras21_cancer1.py를 다중분류로 코딩하시오

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

#원핫코딩 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

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
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (30,))
dense1 = Dense(20, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(20, activation='relu')(dense4)
outputs = Dense(2, activation='sigmoid')(dense5)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=(30,)))
# model.add(Dense(1, activation='sigmoid'))


#3. 컴파일,
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
#이진분류 일때는 무조건 binary_crossentropy
#model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto')
hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 7 ,validation_data = (x_val, y_val), verbose = 1, callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[-5:-1])
print(y_train[-5:-1])
print(y_pred)

#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = 1))

#그래프 출력

print(hist)
print(hist.history.keys()) #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('[cancer_Dense] loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['trian loss', 'val loss', 'train acc', 'val acc'])
plt.show()






#실습1. acc 0.985이상 올릴 것
#실습2. predict 출력해볼것
#y[-5:-1] = ??

# loss :  0.16352708637714386
# acc :  0.9912280440330505
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [1. 0.]]
# [[1.32189485e-11 1.00000000e+00]
#  [9.99997735e-01 2.32061802e-06]
#  [3.37641554e-10 1.00000000e+00]
#  [1.00000000e+00 3.01527137e-10]]
# [1 0 1 0]

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

        The mean, standard error, and "worst" or largest (mean of the three
        worst/largest values) of these features were computed for each image,
        resulting in 30 features.  For instance, field 0 is Mean Radius, field
        10 is Radius SE, field 20 is Worst Radius.
        평균, 표준 오차 및 "최악"또는 최대 (3 개 중 평균
        이러한 특징 중 최악 / 최대 값)은 각 이미지에 대해 계산되었습니다.
        결과적으로 30 개의 기능이 있습니다 예를 들어, 필드 0은 평균 반경, 필드
        10은 반경 SE이고 필드 20은 최악의 반경입니다.
        - class:
                - WDBC-Malignant
                - WDBC-Benign

 '''