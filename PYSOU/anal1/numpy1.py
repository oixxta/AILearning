#기본 통계 함수를 직접 작성하기! : 평균, 분산, 표준편차 구하기
grades = [1, 3, -2, 4]      #표본들

def gradesSum(grades):      #모든 표본들의 함들 구하는 함수
    tot = 0
    for g in grades:
        tot += g
    return tot

print(gradesSum(grades))

def gradesAve(grades):      #모든 표본들의 평균을 구하는 함수
    ave = gradesSum(grades) / len(grades)
    return ave

print(gradesAve(grades))

def gradesVariance(grades): #모든 데이터의 분산값을 구하는 함수
    ave = gradesAve(grades)
    vari = 0
    for su in grades:
        vari += (su - ave)**2
    return vari / len(grades)

print(gradesVariance(grades))

def gradesStd(grades):      #모든 데이터의 표준편차를 구하는 함수
    return gradesVariance(grades) ** 0.5

print(gradesStd(grades))


print('**' * 10)
#미리 만들어진 매서드로 위의 결과물들 구현가능! 따라서 위처럼 직접 구현할 필요 없음!
#그러나 개념은 숙지해야 함.
import numpy as np
print('합은', np.sum(grades))
print('평균은', np.mean(grades))    #산술평균용
print('평균은', np.average(grades)) #가중평균용
print('분산은', np.var(grades))
print('표준편차는', np.std(grades))


