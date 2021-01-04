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
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size = 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (30,))
dense1 = Dense(20, activation='relu')(input1)
dense2 = Dense(20, activation='relu')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(20, activation='relu')(dense4)
outputs = Dense(2, activation='softmax')(dense5)
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
model.fit(x_train, y_train, epochs = 1000, batch_size = 7 ,validation_data = (x_val, y_val), callbacks = [early_stopping])

#4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_train[-5:-1])
print(y_train[-5:-1])
print(y_pred)

#print(np.argmax(y_pred, axis = 0))
print(np.argmax(y_pred, axis = 1))

#실습1. acc 0.985이상 올릴 것
#실습2. predict 출력해볼것
#y[-5:-1] = ??

# loss :  0.1683250069618225
# acc :  0.9890109896659851
# [[0. 1.]
#  [0. 1.]
#  [1. 0.]
#  [0. 1.]]
# [[7.1000011e-07 9.9999928e-01]
#  [8.5848988e-05 9.9991417e-01]
#  [1.0000000e+00 0.0000000e+00]
#  [1.6419766e-04 9.9983573e-01]]
# [1 1 0 1]
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

    :Summary Statistics:

    ===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163  
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
    ===================================== ====== ======

    :Missing Attribute Values: None

    :Class Distribution: 212 - Malignant, 357 - Benign

    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

    :Donor: Nick Street

    :Date: November, 1995

This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
https://goo.gl/U2Uwz2

Features are computed from a digitized image of a fine needle
aspirate (FNA) of a breast mass.  They describe
characteristics of the cell nuclei present in the image.

Separating plane described above was obtained using
Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
Construction Via Linear Programming." Proceedings of the 4th
Midwest Artificial Intelligence and Cognitive Science Society,
pp. 97-101, 1992], a classification method which uses linear
programming to construct a decision tree.  Relevant features
were selected using an exhaustive search in the space of 1-4
features and 1-3 separating planes.

The actual linear program used to obtain the separating plane
in the 3-dimensional space is that described in:
[K. P. Bennett and O. L. Mangasarian: "Robust Linear
Programming Discrimination of Two Linearly Inseparable Sets",
Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:

ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

이것은 UCI ML 유방암 위스콘신 (진단) 데이터 세트의 사본입니다.
https://goo.gl/U2Uwz2

미세 바늘의 디지털화 된 이미지에서 특징 계산
유방 질량의 흡인 (FNA). 그들은 설명합니다
이미지에 존재하는 세포 핵의 특성.

위에서 설명한 분리 평면은
다중 표면 방법 트리 (MSM-T) [K. P. Bennett, "의사 결정 트리
선형 프로그래밍을 통한 구성. "Proceedings of the 4th
중서부 인공 지능 및인지 과학 학회,
pp. 97-101, 1992], 선형을 사용하는 분류 방법
의사 결정 트리를 구성하는 프로그래밍. 관련 기능
1-4의 공간에서 철저한 검색을 사용하여 선택되었습니다.
기능 및 1-3 개의 분리 평면.

분리 평면을 얻는 데 사용되는 실제 선형 프로그램
3 차원 공간에서는 다음에 설명되어 있습니다.
[케이. P. Bennett 및 O. L. Mangasarian : "강력한 선형
선형 적으로 분리 할 수없는 두 세트의 프로그래밍 차별 ",
최적화 방법 및 소프트웨어 1, 1992, 23-34].

이 데이터베이스는 UW CS ftp 서버를 통해서도 사용할 수 있습니다.
.. topic:: References

   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction
     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on
     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
     San Jose, CA, 1993.
   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and
     prognosis via linear programming. Operations Research, 43(4), pages 570-577,
     July-August 1995.
   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994)
     163-171.

       -W.N. 스트리트, W.H. Wolberg 및 O.L. Mangasarian. 핵 특징 추출
     유방 종양 진단을 위해. IS & T / SPIE 1993 국제 심포지엄
     전자 이미징 : 과학 및 기술, 1905 권, 861-870 페이지,
     1993 년 캘리포니아 주 산호세.
   -O.L. Mangasarian, W.N. Street 및 W.H. Wolberg. 유방암 진단 및
     선형 프로그래밍을 통한 예후. 운영 연구, 43 (4), 570-577 페이지,
     1995 년 7 월 -8 월.
   -W.H. Wolberg, W.N. Street 및 O.L. Mangasarian. 기계 학습 기술
     미세 바늘 흡인으로 유방암을 진단합니다. 암 편지 77 (1994)
     163-171.
    '''