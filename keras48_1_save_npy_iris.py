from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)
#{'data': array([[5.1, 3.5, 1.4, 0.2],[4.9, 3. , 1.4, 0.2],
# 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

print(dataset.keys()) #데이터 셋에서 키값 뽑아내기
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

#일반
# x = dataset.data
# y = dataset.target

#딕셔너리 사용
x_data = dataset['data']
y_data = dataset['target']
print(x_data)
print(y_data)

print(dataset.frame) 
#None
print(dataset.target_names)
#y값. ['setosa' 'versicolor' 'virginica']
print(dataset['DESCR'])
#데이터셋 설명
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(dataset.filename)
#C:\ProgramData\Anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv
print(type(x_data), type(y_data))
#<class 'numpy.ndarray'> <class 'numpy.ndarray'>

#데이터 저장***
np.save('../data/npy/iris_x.npy', arr=x_data) #저장은 .npy로 저장할 것
np.save('../data/npy/iris_y.npy', arr=y_data) #iris_를 y_npy로 저장할거고 y_data값을 넣을것이다.

# dataset = load_iris()
# print(dataset)
# print(dataset.keys()) #데이터 셋에서 키값 뽑아내기
# x_data = dataset['data']
# y_data = dataset['target']
# np.save('./data/iris_x.npy', arr=x_data)
# np.save('./data/iris_y.npy', arr=y_data)


