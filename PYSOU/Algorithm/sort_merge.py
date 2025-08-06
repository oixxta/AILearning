#병합정렬(merge sort)
#병합정렬은 분할정복 기법과 재귀 알고리즘을 이용한 정렬 알고리즘.
#주어진 배열을 원소가 하나밖에 남지 않을 때 까지 계속 둘로 쪼갠 후 다시 크기 순으로 재배열 하면서

#흐름 : 리스트를 반으로 쪼갬
#각각 다시 재귀호출로 쪼개기를 계속함
#작은 값부터 병합을 계속함.

#병합 정렬이 선택과 삽입보다 훨씬 더 빠르기에 빅테이터 처리에 유리함!


#방법 1 : 원리이해 우선(공간복잡도 안좋음, 빈 리스트를 한개 만듬.)
def mergeSort(inputList):
    n = len(inputList)
    if(n <= 1):
        return inputList
    mid = n // 2
    group1 = mergeSort(inputList[:mid]) #재귀호출로 첫 번째 그룹(왼쪽 절반)을 정렬
    #print(group1)
    group2 = mergeSort(inputList[mid:]) #재귀호출로 두 번째 그룹(오른쪽 절반)을 정렬
    #print(group2)

    #두 그룹을 하나로 다시 합침
    result = []     #합친 결과 최종 기억
    while group1 and group2:        #두 그룹의 요소값이 있는 동안 반복
        if group1[0] < group2[0]:   #양 그룹의 맨 앞 인덱스의 값을 비교
            result.append(group1.pop(0))
        else:
            result.append(group2.pop(0))
    while group1:                   #두 그룹 중 g1이 남은경우
        result.append(group1.pop(0))
    while group2:                   #두 그룹 중 g2가 남은경우
        result.append(group2.pop(0))

    return result

d = [6, 8, 3, 1, 2, 4, 7, 5]
print(mergeSort(d))

print('-' * 50)

#방법 2 : 공간복잡도도 신경쓰는 방식
def mergeSort2(myList):
    n = len(myList)
    if(n <= 1):
        return
    mid = n // 2
    group1 = myList[:mid]
    group2 = myList[mid:]
    mergeSort2(group1)  #계속 반으로 나누다가 길이가 1이 되면 쪼개기 멈춤
    mergeSort2(group2)
    #두 그룹을 하나하나 합치기
    i1 = 0
    i2 = 0
    ia = 0
    while(i1 < len(group1) and i2 < len(group2)):
        if group1[i1] < group2[i2]: # 두 집합의 앞쪽 값들을 하나씩 비교해 더 작은 것을 a에 차례로 채우기
            myList[ia] = group1[i1]
            i1 += 1
            ia += 1
        else:
            myList[ia] = group2[i2]
            i2 += 1
            ia += 1

    #아직 남아있는 자료들을 추가
    while(i1 < len(group1)):
        myList[ia] = group1[i1]
        i1 += 1
        ia += 1
    while(i2 < len(group2)):
        myList[ia] = group2[i2]
        i2 += 1
        ia += 1

f = [6, 8, 3, 1, 2, 4, 7, 5]
mergeSort2(f)
print(f)


print('두번째 방법을 값을 반환하는 방법으로 변환')
def mergeSort3(myList):
    if len(myList) <= 1:
        return myList
    
    mid = len(myList) // 2
    left = mergeSort3(myList[:mid])
    right = mergeSort3(myList[mid:])
    result = []
    i = j = 0

    # 병합
    while i < len(left) and j < len(right):
        if(left[i] < right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # 남은것 처러
    result += left[i:]
    result += right[j:]

    return result

g = [6, 8, 3, 1, 2, 4, 7, 5]
print(mergeSort3(g))