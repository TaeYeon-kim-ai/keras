import pandas as pd

df = pd.DataFrame([[1,2,3,4], [4,5,6,7], [7,8,9,10]], 
        columns = list('abcd'), index = ('가', '나', '다'))

print(df)

#    a  b  c   d
# 가  1  2  3   4
# 나  4  5  6   7
# 다  7  8  9  10

df2 = df #같은 메모리 공유

df2['a'] = 100
#df2와 df를 나눈것처럼 보이나 같은메모리를 공유함.
print(df)
#  a  b  c   d
# 가  1  2  3   4
# 나  4  5  6   7
# 다  7  8  9  10
print(df2)
#     a  b  c   d
# 가  100  2  3   4
# 나  100  5  6   7
# 다  100  8  9  10
#      a  b  c   d
# 가  100  2  3   4
# 나  100  5  6   7
# 다  100  8  9  10

print(id(df), id(df2))

df3 = df.copy()
df2['b'] = 333
print(df)
print(df2)
print(df3)
#df와 df2가 같은데이터를 공유하므로 df3를 .copy를 활용하여 생성
#     a    b  c   d
# 가  100  333  3   4
# 나  100  333  6   7
# 다  100  333  9  10
#      a    b  c   d
# 가  100  333  3   4
# 나  100  333  6   7
# 다  100  333  9  10
#      a  b  c   d
# 가  100  2  3   4
# 나  100  5  6   7
# 다  100  8  9  10

df = df +99 
print(df)
print(df2)
#덧셈을 했을 시에는 그냥 형성됨
#      a    b    c    d
# 가  199  432  102  103
# 나  199  432  105  106
# 다  199  432  108  109
#      a    b  c   d
# 가  100  333  3   4
# 나  100  333  6   7
# 다  100  333  9  10