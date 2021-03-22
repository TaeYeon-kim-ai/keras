import numpy as np

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#optimizer
# optimizer = Adam(lr = 0.1)
# loss :  0.0643017366528511 결과물 :  [[10.897604]]
# optimizer = Adam(lr = 0.01)
# loss :  0.021259430795907974 결과물 :  [[11.232889]]
# optimizer = Adam(lr = 0.001)
# loss :  7.347011996179653e-13 결과물 :  [[11.000003]]
# optimizer = Adam(lr = 0.0001)
# loss :  4.954847554472508e-06 결과물 :  [[11.001676]]

#Adadelta
#optimizer = Adadelta(lr = 0.1)
#loss :  0.04635532572865486 결과물 :  [[10.635171]]
# optimizer = Adadelta(lr = 0.01)
#loss :  0.0001700993161648512 결과물 :  [[10.974168]]
# optimizer = Adadelta(lr = 0.001)
# loss :  9.002021789550781 결과물 :  [[5.601669]]
# optimizer = Adadelta(lr = 0.0001)
# loss :  40.16439437866211 결과물 :  [[-0.24360675]]

#Adamax
# optimizer = Adamax(lr = 0.1)
#loss :  7.644379615783691 결과물 :  [[16.832827]]
# optimizer = Adamax(lr = 0.01)
# loss :  4.398198250044061e-09 결과물 :  [[11.000087]]
# optimizer = Adamax(lr = 0.001)
# loss :  3.649347317391527e-12 결과물 :  [[11.000001]]
# optimizer = Adamax(lr = 0.0001)
# loss :  0.002011224627494812 결과물 :  [[10.93755]]

#Adagrad
# optimizer = Adagrad(lr = 0.1)
# loss :  1036271.1875 결과물 :  [[1438.961]]
# optimizer = Adagrad(lr = 0.01)
# loss :  8.2372880569892e-06 결과물 :  [[11.003131]]
# optimizer = Adagrad(lr = 0.001)
# loss :  3.211620423826389e-06 결과물 :  [[10.998428]]
# optimizer = Adagrad(lr = 0.0001)
# loss :  0.00583252776414156 결과물 :  [[10.905012]]

#RMSprop
# optimizer = RMSprop(lr = 0.1)
# loss :  15529374720.0 결과물 :  [[-236215.95]]
# optimizer = RMSprop(lr = 0.01)
# loss :  22.316503524780273 결과물 :  [[1.7790612]]
# optimizer = RMSprop(lr = 0.001)
# loss :  0.06729690730571747 결과물 :  [[10.543881]]
# optimizer = RMSprop(lr = 0.0001)
# loss :  0.4086088240146637 결과물 :  [[9.859858]]

#SGD
# optimizer = SGD(lr = 0.1)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr = 0.01)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr = 0.1)
# optimizer = SGD(lr = 0.001)
# loss :  6.13895778656115e-08 결과물 :  [[10.999895]]
# optimizer = SGD(lr = 0.0001)
# loss :  0.0020600378047674894 결과물 :  [[10.942872]]

#Nadam
# optimizer = Nadam(lr = 0.1)
# loss :  0.7213953733444214 결과물 :  [[12.67894]]
# optimizer = Nadam(lr = 0.01)
# loss :  1.5817044973373413 결과물 :  [[9.51435]]
# optimizer = Nadam(lr = 0.001)
# loss :  2.829097026091354e-11 결과물 :  [[10.999991]]
optimizer = Nadam(lr = 0.0001)
# loss :  4.196825102553703e-06 결과물 :  [[10.998614]]

model.compile(loss = 'mse', optimizer = optimizer, metrics=['mse'])
model.fit(x, y, epochs= 100, batch_size=1)

#4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print("loss : ", loss, "결과물 : ", y_pred)

