def selection_sort_with_counts(arr):
    compare_cout = 0
    swap_count = 0
    n = len(arr)
    for i in range(0, n - 1):
        minIndex = i
        for j in range(i + 1, n):
            compare_cout += 1
            if arr[j] < arr[minIndex]:
                minIndex = j
                swap_count += 1
        arr[i], arr[minIndex] = arr[minIndex], arr[i]
    print("정렬 결과 : ", arr)
    print("비교 횟수 : ", compare_cout)
    print("교환 횟수 : ", swap_count)

selection_sort_with_counts([64, 25, 12, 22, 11])