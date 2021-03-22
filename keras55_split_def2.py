# x의 가로세로, y의 가로세로

# 모델을 구성하시오
import numpy as np

#1.데이터
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                     [11,12,13,14,15,16,17,18,19,20],
                     [21,22,23,24,25,26,27,28,29,30],
                     [31,32,33,34,35,36,37,38,39,40],
                     [41,42,43,44,45,46,47,48,49,50],
                     [51,52,53,54,55,56,57,58,59,60]])

# print('dataset.shape : ', dataset.shape)

dataset = np.transpose(dataset)
# print(dataset)
# print('dataset.shape : ', dataset.shape)

def split_xy4(dataset, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(dataset)): #데이터 셋 수만큼 for문 돌리기
        x_row_number = i  #x 시작 정의
        x_col_number = i + x_row  #x 종료  정의       
        y_row_number = x_col_number  #y 시작 정의 x 기준
        y_col_number = y_row_number + y_row #y 종료 정의
        
        if y_col_number > len(dataset) :                
            break

        tmp_x = dataset[x_row_number : x_col_number, :x_col] #x 행렬 
        tmp_y = dataset[y_row_number : y_col_number, x_col : x_col + y_col]   # y 행렬
        x.append(tmp_x) 
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy4(dataset, 4, 4, 3, 2)
# print(x, "\n", y)
print(x.shape)#(4, 4, 4)
print(y.shape)#(4, 3, 2)
print(x)
print(y)

#============================================================================================================================

# dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# def split_xy1(dataset, time_steps): #split_xy함수정의 및 매개변수 dataset, timesteps 정의 dataset:자르고자 하는 데이터셋, time_steps 몇개의 컬럼을 자를건지
#     x ,y = list(), list() #리턴해줄  x와 y를 리스트로 정의

#     for i in range(len(dataset)): #데이터 셋 개수만큼 for문 돌리기
#         end_number = i + time_steps #마지막 숫자가 몇인지 정의 1회전 시 I는 0 0+4 = 4 마지막숫자 는 4
#         if end_number > len(dataset) -1: #마지막 숫자가 dataset전체 길이에서 1개 뺀 값보다 크면 for문 정지
#             break
#         tmp_x, tmp_y = dataset[i:end_number], dataset[end_number]
#         x.append(tmp_x)
#         y.append(tmp_y)
#         return np.array(x), np.array(y) #for문 끝나면 x와 y값 반환

# x, y = split_xy1(dataset, 6)
# print(x, "\n", y)


# #========================

# # #0. 함수정의
# def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
#     dataset = []
#     for i in range(len(seq) - size + 1 ): 
#         subset = seq[i : (i + size),0:col]  
#         dataset.append(subset) 
#     print(type(dataset)) 
#     return np.array(dataset) 

# size = 6 #며칠
# col = 7 #열 수
# dataset = split_x(data1,size, col)
# print('dataset:', dataset)