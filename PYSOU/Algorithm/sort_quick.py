#퀵 정렬(quik sort)
#퀵정렬은 빠르게 정렬을 수행하는 알고리즘으로, 배열 중 한가지를 피벗(pivot, 기준점)으로 줌.
#그리고 배열 내부의 모든 값을 검사하면서 피벗보다 크면 오른쪽, 작으면 왼쪽으로 배치
#나뉜 이 두 개의 배열에서 또 피벗을 지정
#위 과정을 반복

#원리
def quickSort(inputList):
    n = len(inputList)
    if(n <= 1):
        return inputList
    #기준값(피벗) 지정
    pivot = inputList[-1]   #제일 마지막 값, 아무 값이나 사실 상관없음.
    group1 = []
    group2 = []
    for i in range(0, n - 1):
        if(inputList[i] < pivot):
            group1.append(inputList[i])
        else:
            group2.append(inputList[i])
    return quickSort(group1) + [pivot] + quickSort(group2)


a = [6, 8, 3, 1, 2, 4, 7, 5]
print(quickSort(a))


#공간복잡도 최적화
def quickSortSub(inputlist, start, end):
    #종료조건 : 정렬 대상이 한 개 이하이면 정렬할 필요 없음
    if end - start <= 0:
        return 
    pivot = inputlist[end]
    i = start
    for j in range(start, end):
        if inputlist[j] <= pivot:
            inputlist[i], inputlist[j] = inputlist[j], inputlist[i]     #자리바꿈
            i += 1
    inputlist[i], inputlist[end] = inputlist[end], inputlist[i]
    #재귀 실시
    quickSortSub(inputlist, start, i - 1)   #왼쪽부분 정렬
    quickSortSub(inputlist, i + 1, end)

def quickSort2(inputList):
    quickSortSub(inputList, 0, len(inputList) - 1)


b = [6, 8, 3, 1, 2, 4, 7, 5]
quickSort2(b)
print(b)