import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K #텐서플로 백엔드로 돌리겠다.

#함수정의(커스텀 loss)
def custom_mse(y_true, y_pred) :    
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
            #y_true(실제값)에서 y_pred(예측값) 뺀것을 제곱하여 평균으로 나누다(mse)

def quantile_loss(y_true, y_pred) : 
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
    q = tf.constant(np.array([qs]), dtype = tf.float32) #qs를 텐서플로의 (컨스턴트형식) 상수(바뀌지 않는값)형태로 바꾸겠다. tf 1.x
                                                        #tf.constant 검색                
    e = y_true = y_pred #y_true(실제값)에서 y_pred(예측값)
    v = tf.maximum(q*e, (q-1)*e) 
    return K.mean(v)

#데이콘 버전 quantile loss ,quantiles를 밖으로 뺀 버전
def quantile_loss_dacon(q, y_true, y_pred): 
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


#====================================================

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
model.compile(loss = quantile_loss , optimizer='adam')
                     #함수로 작성한 mse를 자의적으로 넣어줌
#데이콘 compile
model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles[0], y_true, y_pred), optimizer='adam')
                #람다값 y_true, y_pred                              #0.1
                #0~9까지 
model.fit(x, y, batch_size=1, epochs=50)

loss = model.evaluate(x, y)
print(loss)
aaa = model.predict(x)
print(aaa)

#custom mse
#0.0013026794185861945

#quantile_loss
#0.0034402902238070965

#quantile_loss_dacon 0.1값
# 0.002166897989809513


# def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
#     # (a) Modeling  #회귀모델
#     model = LGBMRegressor(objective='quantile', alpha=q, #LGBM에서 quantile 지정 0.1 ~ 0.9 넣으면 quantile loss 끝
#                          n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
#     model.fit(X_train, Y_train, eval_metric = ['quantile'], 
#           eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
