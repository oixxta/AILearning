import numpy as np

# 1) step1 : array 관련 문제
# 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.
print("-" * 30)
print("문제 1")

myArray = np.full((5, 4), fill_value = np.random.randn())
print(myArray)
totalValue = np.sum(myArray)
print(f"합계 : {totalValue}")
maxValue = np.amax(myArray)
print(f"최댓값 : {maxValue}")

# 2) step2 : indexing 관련문제
# 문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.
print("-" * 30)
print("문제 2-1")

myArray = np.zeros((6, 6))  #제로행렬 생성
print(myArray)

inputInt = 1                #36개의 셀에 1~36까지 정수 채우기
for i in range(6):
    for j in range(6):
        myArray[i][j] = inputInt
        inputInt += 1
print(myArray)
print(myArray[1:2])         #2번째 행 전체 원소 출력하기
print(myArray[:, 5:6])      #5번째 열 전체 원소 출력하기
print(myArray[2:5, 2:5])    #15~29 까지 아래 처럼 출력하기


# 문2-2) 6행 4열의 다차원 zero 행렬 객체를 생성한 후 아래와 같이 처리하시오.
print("-" * 30)
print("문제 2-2")

myArray = np.zeros((6,4))   #제로행렬 생성
print(myArray)
#20~100 사이의 난수 정수를 6개 발생시켜 각 행의 시작열에 난수 정수를 저장하고, 
#두 번째 열부터는 1씩 증가시켜 원소 저장하기
import random
for i in range(6):
    for j in range(4):
        if(j == 0):
            myArray[i][j] = random.randint(20, 100)
        else:
            myArray[i][j] = myArray[i][j-1] + 1

print(myArray)

#첫 번째 행에 1000, 마지막 행에 6000으로 요소값 수정하기
for i in range(6):
    for j in range(4):
        if(i == 0):
            myArray[i][j] = 1000
        elif(i == 5):
            myArray[i][j] = 6000
        else:
            pass

print(myArray)


# step3 : unifunc 관련문제
#표준정규분포를 따르는 난수를 이용하여 4행 5열 구조의 다차원 배열을 생성한 후
#아래와 같이 넘파이 내장함수(유니버설 함수)를 이용하여 기술통계량을 구하시오.
#배열 요소의 누적합을 출력하시오.
print("-" * 30)
print("문제 3")

myArray3 = np.random.randn(4, 5)
print(myArray3)
print(f"평균 : {np.mean(myArray3)}")
print(f"합계 : {np.sum(myArray3)}")
print(f"표준편차 : {np.std(myArray3)}")
print(f"분산 : {np.var(myArray3)}")
print(f"최댓값 : {np.amax(myArray3)}")
print(f"최솟값 : {np.amin(myArray3)}")
print(f"1사분위 수 : {np.percentile(myArray3, 25)}")
print(f"2사분위 수 : {np.median(myArray3)}")
print(f"3사분위 수 : {np.percentile(myArray3, 75)}")
print(f"요소값 누적합 : {np.cumsum(myArray3)}")



# numpy 문제 추가 : 브로드캐스팅과 조건 연산
# 다음 두 배열이 있을 때, 두 배열을 브로드캐스팅하여 곱한 결과를 출력하시오.
# 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.
print("-" * 30)
print("추가문제 1")


# numpy 문제 추가 : 다차원 배열 슬라이싱 및 재배열
# 3×4 크기의 배열을 만들고 (reshape 사용), 2번째 행 전체 출력, 1번째 열 전체 출력
# 배열을 (4, 3) 형태로 reshape, reshape한 배열을 flatten() 함수를 사용하여 1차원 배열로 만들기



# numpy 문제 추가 :
# 1부터 100까지의 수로 구성된 배열에서 3의 배수이면서 5의 배수가 아닌 값만 추출하시오.
# 그런 값들을 모두 제곱한 배열을 만들고 출력하시오.



# numpy 문제 추가 :
# 값이 10 이상이면 'High', 그렇지 않으면 'Low'라는 문자열 배열로 변환하시오.
# 값이 20 이상인 요소만 -1로 바꾼 새로운 배열을 만들어 출력하시오. (원본은 유지)



# numpy 문제 추가 :
# 정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.
# 힌트 :  np.random.normal(), np.percentile()




