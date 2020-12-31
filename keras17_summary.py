import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성(인공신경망)
model = Sequential() #순차적 모델
model.add(Dense(5, input_dim=1, activation='linear',)) #dam 1개 차원
model.add(Dense(30, activation='linear')) # 모델 수 hidden layer
model.add(Dense(30, na))# 모델 수 hidden layer 
model.add(Dense(30))# 모델 수 hidden layer
model.add(Dense(30))# 모델 수 hidden layer
model.add(Dense(40, name))# 모델 수 hidden layer
model.add(Dense(40))# 모델 수 hidden layer
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 30)                180
_________________________________________________________________
dense_2 (Dense)              (None, 30)                930
_________________________________________________________________
dense_3 (Dense)              (None, 30)                930
_________________________________________________________________
dense_4 (Dense)              (None, 30)                930
_________________________________________________________________
dense_5 (Dense)              (None, 40)                1240
_________________________________________________________________
dense_6 (Dense)              (None, 40)                1640
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 41
=================================================================
Total params: 5,901
Trainable params: 5,901
Non-trainable params: 0
'''

# 실습2 + 과제
#ensemble 1, 2, 3, 4에 대해 서머리를 계산하고 이해한 것을 과제로 제출할 것
#layer을 만들 때 'name'에 대해 확인하고 설명할 것 name을 반드시 써야할 때가 있나?

