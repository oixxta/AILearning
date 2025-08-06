#삽입정렬(insertion sort)
#삽입정렬은 자료 배열의 모든 요소를 앞에서부터 차례대로 이미 정렬된 배열 부분과 비교해 
#자신의 위치를 찾아 삽입함으로서 정렬을 완성하는 알고리즘

#방법 1 : 원리이해 우선
def findInsFunc(r, v):
    #이미 정렬된 r의 자료를 앞에서 부터 차례대로 확인
    for i in range(0, len(r)):
        if (v < r[i]):
            return i
    return len(r)   #v가 r의 모든 요소값보다 클 경우에는 맨 뒤에 삽입!

def insSort(a):
    result = []
    while a:
        value = a.pop(0)
        insIndex = findInsFunc(result, value)
        result.insert(insIndex, value)  #찾은 위치에 값을 삽입(이후 값은 밀려남) 또는 추가
        print(result)
    return result

d = [2, 4, 5, 1, 3, 10]
print(insSort(d))
print('-' * 50)



#방법 2 : 공간복잡도도 신경쓰는 방식

def insSort2(a):
    n = len(a)
    for i in range(1, n):   #두번째 값(인덱스1)부터 마지막까지 차례대로 '삽입할 대상' 선택
        key = a[i]
        j = i - 1
        while(j >= 0 and a[j] > key):   #key값보다 큰 값을 우측으로 밀기(참)
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
        print(a)
    

f = [2, 4, 5, 1, 3]
insSort2(f)
print(f)
