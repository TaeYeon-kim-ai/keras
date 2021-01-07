'''
#모델 save를 위한 코드 별도 선언 없음
# 1. 작업영역에 model(저장할 폴더 위치) 생성
# 2. 작업영역에 model(저장할 폴더 위치) 생성
# 3. 모델 작성
# 4. 저장코드 작성 : model.save("./model/save_keras35.h5) <- 가장 앞의 .은 현 위치에 생성함을 의미
# sub. /, //, \, \\모두 사용할 수 있으나 통일 되어야함.
'''
import numpy as np
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, LSTM, Input

#2. 모델
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (4, 1)))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))

model.summary()

#모델저장
model.save("./model/save_keras35.h5")
model.save(".//model//save_keras35_1.h5")
model.save(".\model\save_keras35_2.h5")
model.save(".\\model/\\save_keras35_3.h5")
