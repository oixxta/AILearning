# Broadcasting : 크기가 다른 배열 간의 연산시 배열의 구조 자동 변환
# 작은배열과 큰 배열 연산 시 작은 배열은 큰 배열에 구조를 따름.

import numpy as np

x = np.arange(1, 10).reshape(3, 3)
y = np.array([1, 0, 1])
print(x)
print(y)

#두 배열의 요소 더하기!
# 1. 새로운 배열을 이용
z = np.empty_like(x)    # x와 같은 구조의 새 배열 z 만듬(내용물은 무작위)
print(z)

for i in range(3):
    z[i] = x[i] + y
print(z)

# 2. tile을 이용
kbs = np.tile(y, (3, 1))
print('kbs : ', kbs)
z = x + kbs
print(z)

# 3. Broadcasting 이용
# 1D + 1D(같은 길이), 1D + 1D(한쪽길이1), 2D + 1D 가능
# 1D + 1D(길이 다르고 1도 아님) 불가능
kbs = x + y
print(kbs)

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print(a + 5)

print('\n 넘파이로 파일 입출력이 가능함!')
np.save('numpy4etc', x)             #2진수 형식으로 저장
np.savetxt('numpy4etc.txt', x)
imsi = np.load('numpy4etc.npy')
print(imsi)

mydatas = np.loadtxt('numpy4etc2.txt', delimiter=',')   #파일 읽어오기, 델미터는 구분자
print(mydatas)

