#선택정렬(selection sort)
#선택정렬은 주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로 이동하게 한 다음

#알고리즘 과정:
#최소값 찾기: 정렬되지 않은 부분에서 가장 작은 값 찾기
#교환: 찾은 최소값을 정렬되지 않은 부분의 맨 앞으로 이동
#반복: 정렬되지 않은 부분의 크기가 1이 될 때까지 위 과정을 반복

#방법 1 : 원리이해 우선(공간복잡도 안좋음, 빈 리스트를 한개 만듬.)
def findMinFunc(a):
    n = len(a)
    minIdx = 0
    for i in range(1, n):
        if(a[i] < a[minIdx]):
            minIdx = i
    return minIdx

def selectionSort(a):
    resultList = []
    while a:
        minIndex = findMinFunc(a)
        value = a.pop(minIndex)
        resultList.append(value)
    return resultList

d = [2, 4, 5, 1, 3]
print(selectionSort(d))


#방법 2 : 공간복잡도도 신경쓰는 방식
#각 반복마다 가장 작은 값을 해당 집합 내의 맨 앞자리와 값을 바꿈!(새 리스트 안만듬)
def selectionSort2(myList):
    n = len(myList)
    for i in range(0, n - 1):   #0부터 n - 2까지 반복
        minIndex = i
        for j in range(i + 1, n):
            if myList[j] < myList[minIndex]:
                minIndex = j
        myList[i], myList[minIndex] = myList[minIndex], myList[i]

f = [2, 4, 5, 1, 3, 10]
selectionSort2(f)
print(f)
