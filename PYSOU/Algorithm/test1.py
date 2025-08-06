#<알고리즘>
#문제를 해결하기 위한 일련의 단계적 절차 또는 방법
#어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어를 순서대로 나열한 것이라 할 수 있음.
#컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 한다.
#'문제 -> 데이터입력 -> 알고리즘으로 처리 -> 결과 출력'이 알고리즘의 기본



#문1) 1 ~ 10(n) 까지의 연속된 정수의 합 구하기
def totFunc(n):     #방법 1 -> O표기법으로는 시간복잡도 O(n)
    tot = 0
    for i in range(1, n + 1):
        tot = tot + i
    return tot

print(totFunc(100))

def totFunc2(n):    #방법 2 -> O표기법으로는 시간복잡도 O(1), 방법 1보다 더 빠름
    return n * (n + 1) // 2     #덧셈 후 곱셈 후 나눗셈

print(totFunc2(100))

#주어진 문제를 푸는 방법은 다양하다. 어떤 방법이 더 효과적인지 알아내는 것이 '알고리즘 분석'
#'알고리즘 분석' 평가 방법으로 계산 복잡도 표현 방식이 있음.
# 1) 공간 복잡도 : 메모리 사용량 분석
# 2) 시간 복잡도 : 처리 시간을 분석
# O(빅 오) 표기법 : 알고리즘의 효율성을 표현해주는 표기법


#문2) 임의 정수들 중 최대값 찾기
#입력 : 숫자 n개를 가진 list
#출력 : 숫자 n개 중 최대값을 출력
def findMaxFunc(a):                 #방법1) 시간복잡도 O(n)
    maxValue = a[0]
    for i in range(1, len(a)):
        if (a[i] > maxValue):
            maxValue = a[i]
    return maxValue

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc(d))

#최대값 위치(인덱스) 반환
def findMaxFunc2(a):                 #방법2) 시간복잡도 O(n)
    maxValue = 0
    for i in range(1, len(a)):
        if (a[i] > a[maxValue]):
            maxValue = i
    return maxValue

d = [17, 92, 11, 33, 55, 7, 27, 42]
print(findMaxFunc2(d))


#문3) 동명이인 찾기 : n명의 사람 이름 중 동일한 이름을 찾아 결과를 출력
imsi = ['길동', '순식', '순식', '길동']
imsi2 = set(imsi)
imsi = list(imsi2)
print(imsi)

def findSameFunc(a):                #시간복잡도 O(n * n)
    n = len(a)
    result = set()
    for i in range(0, n-1): #0부터 n-2까지 반복
        for j in range(i + 1, n):
            if(a[i] == a[j]):           #이름이 같으면
                result.add(a[i])
    return result

names = ['tom', 'jerrt', 'mike', 'tom', 'tom', 'mike']
print(findSameFunc(names))


#문4) 팩토리얼 구하는 알고리즘
#재귀함수 : 자기 자신을 호출하는 함수, 종료조건이 필수로 정의되어야 함 + 분할정복이 사용되어야 함
# 방법 1 - 반복문 사용
def factirialFunc(n):
    imsi = 1
    for i in range(1, n + 1):
        imsi = imsi * i
    return imsi

print(factirialFunc(5))

# 방법 2 - 재귀 사용
def factoricalFunc2(n):
    if(n <= 1):         #종료 조건
        return 1
    return n * factoricalFunc2(n - 1)   #재귀 호출

print(factoricalFunc2(5))


# 재귀 연습1) 1부터 n까지의 합 구하기 : 재귀를 써서(반복문 x)
def recurtionFunc1(n):
    if(n <= 1):
        return 1
    return n + recurtionFunc1(n - 1)
print("문제 1 정답 : ", recurtionFunc1(3))


# 재귀 연습2) 숫자 n개 중 최대값 구하기 : 재귀를 써서(반복문 x)
def recurtionFunc2(a, n):
    if(n - 1 == 0):          #종료조건
        return a[0]
    
    if(a[n - 1] > a[n - 2]):
        a.pop(n - 2)
    else:
        a.pop(n - 1)
    return recurtionFunc2(a, n - 1)

values = [7, 9, 15, 42, 33, 22]

print("문제 2 정답 : ", recurtionFunc2(values, len(values)))