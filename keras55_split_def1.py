#split_xy
import numpy as np
#1.데이터
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
                     [11,12,13,14,15,16,17,18,19,20],
                     [21,22,23,24,25,26,27,28,29,30],
                     [31,32,33,34,35,36,37,38,39,40],
                     [41,42,43,44,45,46,47,48,49,50],
                     [51,52,53,54,55,56,57,58,59,60]])
# #0. 함수정의
def split_x(seq, size, col):  #size = column #col은 여러개 변수로 예측을 하기때문에 사용
    dataset = []
    for i in range(len(seq) - size + 1 ): 
        subset = seq[i : (i + size),0:col]  
        dataset.append(subset) 
    print(type(dataset)) 
    return np.array(dataset) 

size = 6 #며칠
col =6 #열 수
dataset = split_x(dataset,size, col)
print('dataset:', dataset)

x = dataset[3,3,3]
y = dataset[1:,0,-1] # OR [1:,-1:,0] 하루 건너 치, y=df[1:, -2:, 0]이틀치 from.주형
print(x.shape)
print(y.shape)

#==============================================================
# # x의 가로세로, y의 가로세로

# # 모델을 구성하시오
# import numpy as np

# #1. 데이터
# a = np.array(range(1, 11))
# size = 5

# def split_x(seq, size): 
#     aaa = []
#     for i in range(len(seq) - size + 1 ): #for반복문 i를 반복해라 size + 1 까지
#         subset = seq[i : (i + size)]  #seq를 구성해라 i(1)부터 i+size(5)까지
#         aaa.append(subset) # aaa에 추가해라 [] 한바퀴돌
#     print(type(aaa)) #aaa 의 타입을 추가해라
#     return np.array(aaa) #aaa를 반환하라

# dataset = split_x(a, size) #dataset에 추가
# print("===========================")
# #print(dataset) #split_x datasets을 0~10까지 size 5까지 순서대로 넣기

# x = dataset[:,0:4]  # [0:6, 0:4] [:,0:4] or [:,0:-2]
# y = dataset[:,4:] # p[0:6, 4] [:,4]or [:,-1]
# #print(x.shape) #  (6, 4)
# #print(y.shape) #  (6, 1)
# print(x) 
# print(y) # [ 5  6  7  8  9 10]

#==============================================================

# # dataset = np.load('./stock_pred/SSD_prepro_data3.npy')
# # time_steps = 5
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

# #====================================================================
# # dataset = np.load('./stock_pred/SSD_prepro_data3.npy')
# # # dataset = np.array([1,2,3,4,5,6,7,8,9,10])

# # def split_xy2(dataset, time_steps, y_column):
# #     x, y = list(), list()
# #     for i in range(len(dataset)):
# #         x_end_number = i + time_steps                      # end_number -> x_end_number 로 수정, x값의 끝번호 i가 0일 때 현재 time_steps가 4이므로 x_end_number은 4 // y_end_number추가, x_end_number이 x의 끝이므로 y_end_number은 y의 개수만큼 추가되어 끝자리를 나타냄
# #         y_end_number = x_end_number + y_column             #추가 y_column매개변수 추가, y값의 원하는 열의 개수를 지정하기 위함
# #         # if_end_number > len(dataset) -1;
# #         #break
# #         if y_end_number > len(dataset) :                    #수정 y_end_number의 끝이 10 이상이 될 경우  for문 중지
# #             break
# #         tmp_x = dataset[i : x_end_number]
# #         tmp_y = dataset[x_end_number : y_end_number]       #수정
# #         x.append(tmp_x)
# #         y.append(tmp_y)

# #     return np.array(x), np.array(y)

# # time_steps = 4
# # y_column = 2
# # x, y = split_xy2(dataset, time_steps, y_column)

# # print(x, "\n", y)
# # # [[1 2 3 4]
# # #  [2 3 4 5]
# # #  [3 4 5 6]
# # #  [4 5 6 7]
# # #  [5 6 7 8]]

# # #   [[ 5  6]
# # #  [ 6  7]
# # #  [ 7  8]
# # #  [ 8  9]
# # #  [ 9 10]]
# # print(x.shape)
# # print(y.shape)
# # (5, 4)
# # (5, 2)

# #==================================================================

# #.split 함수 만들기3 (다입력, 다:1)
# #2차원 이상 데이터 입력 x = 다차원 배열, y=벡터형태

# #1데이터
# # dataset = np.array([[1,2,3,4,5,6,7,8,9,10],
# #  [11,12,13,14,15,16,17,18,19,20],
# #   [21,22,23,24,25,26,27,28,29,30]])

# # # print('dataset.shape : ', dataset.shape)

# # dataset = np.transpose(dataset)
# # # print(dataset)
# # # print('dataset.shape : ', dataset.shape)

# def split_xy3(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps                      # end_number -> x_end_number 로 수정, x값의 끝번호 i가 0일 때 현재 time_steps가 4이므로 x_end_number은 4 // y_end_number추가, x_end_number이 x의 끝이므로 y_end_number은 y의 개수만큼 추가되어 끝자리를 나타냄
#         y_end_number = x_end_number + y_column -1             #추가 y_column매개변수 추가, y값의 원하는 열의 개수를 지정하기 위함
        
#         if y_end_number > len(dataset) :                    #수정 y_end_number의 끝이 10 이상이 될 경우  for문 중지
#             break
#         tmp_x = dataset[i : x_end_number, :-1]
#         tmp_y = dataset[x_end_number-1 : y_end_number, -1]       #수정
#         x.append(tmp_x)
#         y.append(tmp_y)

#     return np.array(x), np.array(y)
# x, y = split_xy3(dataset, 3, 1)
# # print(x, "\n", y)
# print(x.shape)#(8, 3, 2)
# print(y.shape)#(8, 1)
# print(x)




# (8,) : 벡터형태로 변환
# print(y.shape)
# [[[ 1 11]
#   [ 2 12]
#   [ 3 13]]

#  [[ 2 12]
#   [ 3 13]
#   [ 4 14]]

#  [[ 3 13]
#   [ 4 14]
#   [ 5 15]]

#  [[ 4 14]
#   [ 5 15]
#   [ 6 16]]

#  [[ 5 15]
#   [ 6 16]
#   [ 7 17]]

#  [[ 6 16]
#   [ 7 17]
#   [ 8 18]]

#  [[ 7 17]
#   [ 8 18]
#   [ 9 19]]
# print(y)
# # [[23]
# #  [24]
# #  [25]
# #  [26]
# #  [27]
# #  [28]
# #  [29]
# #  [30]]

# y = y.reshape(y.shape[0])
# print(y.shape)
# # (8,)
# print(y)
# [23 24 25 26 27 28 29 30]

# def split_xy(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps                     
#         y_end_number = x_end_number + y_column
    
#         if y_end_number > len(dataset) :      
#             break
#         tmp_x = dataset[i : x_end_number, :]
#         tmp_y = dataset[x_end_number : y_end_number, :]     
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy(train, 6, 2)
# print(x, "\n", y)
# print(x.shape)
# print(y.shape)
