import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5

def split_x(seq, size): 
    aaa = []
    for i in range(len(seq) - size + 1 ): #for반복문 i를 반복해라 size + 1 까지
        subset = seq[i : (i + size)]  #seq를 구성해라 i(1)부터 i+size(5)까지
        aaa.append(subset) # aaa에 추가해라 [] 한바퀴돌
    print(type(aaa)) #aaa 의 타입을 추가해라
    return np.array(aaa) #aaa를 반환하라

dataset = split_x(a, size) #dataset에 추가
print("===========================")
print(dataset.shape) # (96, 5) #split_x datasets을 0~10까지 size 5까지 순서대로 넣기

x = dataset[:, 0:4] #(96, 4)
y = dataset[:, -1] # (96,)
print(x.shape)
print(y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape) # (96, 4, 1)

#2. 모델
model = load_model('./model/save_keras35.h5')
model.add(Dense(5, name = 'kingkeras1'))
model.add(Dense(1, name = 'kingkeras2'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience = 10, mode = 'auto')
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
hist = model.fit(x, y, epochs = 1000, batch_size = 1, verbose = 1, validation_split = 0.2, callbacks = [es])

print(hist)
print(hist.history.keys()) #'loss', 'acc', 'val_loss', 'val_acc'

print(hist.history['loss']) #변수가 하나하나 보임 1eposh 당 변화되는 loss값을 출력해줌  히스토리~이력출력

#그래프
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc') #y축 라벨링
plt.xlabel('epoch') #x축 라벨링
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 지표
plt.show()

#val loss와 train loss가 최대한 붙어있어야 검증이 잘되어있다는 뜻임.
#loss와 val_loss(검증)이 차이가 많이 나면 과적합
#loss와 val_loss(검증)이 차이가 적을 수록 검증이 잘 된 모델

'''
cancer
wine
iris
'''