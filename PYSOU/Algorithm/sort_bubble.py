#버블 정렬(bubble sort)
#버블정렬은 인접한 두 개의 요소를 비교, 그리고 자리를 교환하는 것이 전부.
#가장 시간복잡도가 크기 때문에 별로 좋지 않음.

def bubbleSort(myList):
    n = len(myList)
    while True:
        changed = False     #반복문 중단 플래그
        for i in range(0, n - 1):
            if(myList[i] > myList[i + 1]):
                print(myList)
                myList[i], myList[i + 1] = myList[i + 1], myList[i]
                changed = True
        if changed == False:
            return


b = [2, 4, 5, 1, 3]
bubbleSort(b)
print(b)