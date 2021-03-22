import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

#함수정의(커스텀 loss)
def custom_mse(y_true, y_pred) :    
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
            #y_true(실제값)에서 y_pred(예측값) 뺀것을 제곱하여 평균으로 나누다(mse)

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') #custom_mse가 int형을 먹지 않으므로 실수형으로 변환
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

print(x.shape) #(8,)
print(y.shape) #(8,)

#2. 모델
model = Sequential()
model.add(Dense(10, input_shape = (1,)))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = custom_mse , optimizer='adam')
                     #함수로 작성한 mse를 자의적으로 넣어줌
model.fit(x, y, batch_size=1, epochs=50)

loss = model.evaluate(x, y)

print(loss)
