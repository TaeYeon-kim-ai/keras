#실습 validation_data  를 만들것!!!
#train_test_split를 사용할 것.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,101))   #w = 1 b = 100
y = np.array(range(1,101))

#x_train = x[:60]   # 순서 0부터 59번째 까지 :::: 값 1 ~ 60
#x_val = x[60:80]   # 61 ~ 80
#x_test = x[80:] # 81 ~ 100
# 리스트 슬라이싱

#y_train = y[:60]   # 순서 0부터 59번째 까지 :::: 값 1 ~ 60
#y_val = y[60:80]   # 61 ~ 80
#y_test = y[80:] # 81 ~ 100
#리스트 슬라이싱

from sklearn.model_selection import train_test_split #사이킷런 사용
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9 ,test_size=0.2, shuffle=True)
                                                    #train_size=0.9 ,test_size=0.2, shuffle=False) #train과 test를 6:4로 분류
                                                    #train_size=0.7, test_size=0.2, shuffle=True)
                                                    #위 두가지의 경우에 대해 확인 후 정리할 것!!!
                                                   
                                                    #A: train_size=0.9, test_size=0.2의 경우 > 1 : 오류코드 출력 train_size와 test_size는 1범위 내에 있어야함.
                                                        # ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
                                                    
                                                    #A: train_size=0.7, test_size=0.2의 경우 < 1 : 데이터 10%를 버림
                                                        # [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
                                                        #25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
                                                        #49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70]
                                                        # (70,)
                                                        # (20,)
                                                        # (70,)
                                                        # (20,)

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=False) #train과 test를 6:4로 분류
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(23))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) #validation 은 train data의 20%

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predcit = model.predict(x_test)
print(y_predcit)

#사이킷런이 뭔지 2분 검색 sikit-learn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predcit))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predcit)
print("R2 : ", r2 )

# shuffle = False
#loss :  0.13000360131263733
#mae :  0.35600584745407104

# shuffle = True
#loss :  0.007983502000570297
#mae :  0.06726665794849396

# validation 1 = 0.2
#loss :  0.05742223188281059
#mae :  0.23790398240089417

# validation 2 = 0.2
#loss :  0.0019563143141567707
#mae :  0.03857726976275444
#RMSE : 0.044230673563057835
#R2 :  0.9999979836146844
