#배열의 연산
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x, x.astype, x.dtype)

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.arange(5, 9).reshape(2, 2)
y = y.astype(np.float64)
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

#요소별 합
print(x + y)    #파이썬이 제공하는 산술연산자
print(np.add(x, y)) #넘파이의 유니버설 함수 중 add 함수 사용
# np.subtract, np.multiply, np.divide
import time
big_arr = np.random.rand(1000000)
start = time.time()
sum(big_arr)    #파이썬 내장함수
end = time.time()
print(f"sum(): {end - start:.6f}sec")

start = time.time()
np.sum(big_arr)    #넘파이의 함수
end = time.time()
print(f"np.sum(): {end - start:.6f}sec")    #연산속도 측정 결과, 넘파이가 압도적으로 더 빠름!


#요소별 곱
print(x * y)    #파이썬이 제공하는 산술연산자
print(np.multiply(x, y)) #넘파이의 유니버설 함수 중 multiply 함수 사용


# 내적 연산
v = np.array([9, 10])
w = np.array([11, 12])
print(v * w)
print(v.dot(w)) #내적 연산법 : 9 * 11 + 10 * 12, 파이썬 내장 함수
print(np.dot(v, w)) #내적 연산법, 넘파이 내장, 훨씬 더 빠름.
print()

print(v * w)
print(v.dot(w))
print(np.dot(v, w))
print(np.dot(x, v))

print(np.dot(x, y))

print('유용한 함수 --------')
print(x)
print(np.sum(x, axis = 0))  #열단위 연산
print(np.sum(x, axis = 1))  #행단위 연산

print(np.min(x), ' ', np.max(x))        #값 자체를 리턴
print(np.argmin(x), ' ', np.argmax(x))  #값이 아닌, 배열의 인덱스 값을 리턴
print(np.cumsum(x))     #배열의 누적합
print(np.cumprod(x))    #배열의 누적곱
print()

names = np.array(['tom', 'james', 'oscar', 'tom', 'oscar'])
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names))     #배열의 중복 제거 출력 후 알파벳 순으로 출력
print(np.intersect1d(names, names2))    #두 배열의 교집합 출력(중복 미허용)
print(np.intersect1d(names, names2, assume_unique= True))    #두 배열의 교집합 출력(중복 허용)
print(np.union1d(names, names2))

print('\n 전치(Transpose)')
print(x)
print(x.T)  #두 배열의 요소가 전치됨
arr = np.arange(1, 16).reshape((3, 5))
print(arr)
print(arr.T)
print(np.dot(arr.T, arr))

print(arr.flatten())    #차원 축소
print(arr.ravel())      #차원 축소