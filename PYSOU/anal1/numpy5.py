import numpy as np

#배열에 행 또는 열 추가하기
aa = np.eye(3)
print('aa : \n', aa)
bb = np.c_[aa, aa[2]]       #행 추가
print(bb)
cc = np.r_[aa, [aa[2]]]     #열 추가
print(cc)

# reshape
a = np.array([1,2,3])
print('np.c_:\n', np.c_[a])
a.reshape(3,1)
print(a)

print('--append, insert, delete--')
#1차원 배열 전용
print(a)
b = np.append(a, [4, 5])
print(b)
c = np.insert(a, 2, [6, 7])
print(c)
#d = np.delete(a, 1)
d = np.delete(a, [1, 2])
print(d)
print()

#2차원 배열 전용
aa = np.arange(1, 10).reshape(3, 3)
print(aa)
print(np.insert(aa, 1, 99))     #삽입 후 차원 축소가 이루어짐.
print(np.insert(aa, 1, 99, axis=0)) #행 기준
print(np.insert(aa, 1, 99, axis=1)) #열 기준
print(aa)       #원본은 그대로 보존됨.

bb = np.arange(10, 19).reshape(3,3)
print(bb)
cc = np.append(aa, bb)          #추가 후 차원 축소가 이루어짐.
print(cc)
cc = np.append(aa, bb, axis = 0)#행으로 들어감, 추가 후 차원이 그대로 유지됨.
print(cc)
cc = np.append(aa, bb, axis = 1)#열로 들어감, 추가 후 차원이 그대로 유지됨.
print(cc)

print("np.append 연습!")
print(np.append(aa, [[88, 88, 88]], axis=0))    #행으로 추가
print(np.append(aa, [[88], [88], [88]], axis=1))#열로 추가

print()
print(np.delete(aa, 1)) #첫번째 것 제거, 차원축소 발생
print(np.delete(aa, 1, axis = 0))   #첫번째 것 제거, 차원축소 미발생
print(np.delete(aa, 1, axis = 1))


#조건 연산 where(조건, 참, 거짓)
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
condData = np.array([True, False, True])
result = np.where(condData, x, y)
print(result)

aa = np.where(x >= 2)
print(aa)   #(array([1, 2]),) index
print(x[aa])
print(np.where(x >= 2, 'T', 'F'))
print(np.where(x >= 2, x, x + 100))

bb = np.random.randn(4, 4)  #정규분포(가우시안 분포 - 중심극한정리)를 따르는 난수
print(bb)
print(np.where(bb > 0, 7, bb))  #0을 초과할 경우 7이, 아닐경우 원래값(bb)

print('배열 결합/분할')
kbs = np.concatenate([x, y])    #배열 결합
print(kbs)

x1, x2 = np.split(kbs, 2)   #배열 분할
print(x1)
print(x2)
print()
a = np.arange(1, 17).reshape(4, 4)
print(a)
x1, x2 = np.hsplit(kbs, 2)
print(x1)
print(x2)

print('복원, 비복원 추출(셈플링)')
#데이터 집합에서 무작위의 표본을 뽑아냄
datas = np.array([1, 2, 3, 4, 5, 6, 7])
#복원 추출
for _ in range(5):
    print(datas[np.random.randint(0, len(datas) - 1)], end = ' ')

#비복원 추출 전용 함수 - sample()
print()
import random
print(random.sample(list(datas), 5))

print("-------------------------------------------")
#추출 함수 : choice()
#복원 추출
print(np.random.choice(range(1, 46), 6))
#비복원 추출
print(np.random.choice(range(1, 46), 6, replace=False))
#가중치를 부여한 랜덤 추출
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar, 3, p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])) #7번째가 가중치가 제일 높음

