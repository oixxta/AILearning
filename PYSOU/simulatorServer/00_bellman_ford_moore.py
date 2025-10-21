"""
벨만-포드-무어 알고리즘

플루이드 워셜은 노드사이의 거리 중 음수인 것이 있다면, 해당 길만 무한히 반복하게 됨.
벨만-포드-무어는 음의 가중치를 가지는 간선도 가능하며,

그러나, 벨만 포드 역시 음의 값이 계속 누적되는 사이클에 빠지면 문제가 됨.(의미없는 값 반환)
"""
INF = int(1e9)  # 무한을 의미하는 10억

# 노드의 갯수 입력받기
n, m = map(int, input().split())
edges = []
dis = [INF] * (n + 1)       #최단 거리 테이블

# 각 간선에 대한 정보를 입력받아, 그 값으로 초기화.
for _ in range(m):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))


def bf(start):
    dis[start] = 0  #시작 지점 초기화

    #매 반복 마다 모든 간선 확인
    # 음의 간선 사이클 존재 유무가 필요하다면 n번과 return 처리.
    # 필요 없다면, n-1번과 리턴 처리 필요 없음. dis 테이블만 필요함.
    for i in range(n + 1):
        for j in range(m):
            current = edges[j][0]
            next_node = edges[j][1]
            cost = edges[j][2]
            # 시작 위치에서 현재 노드까지 이동이 가능하면서
            # 현재 간선을 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if dis[current] != INF and dis[next_node] > cost + dis[current]:
                dis[next_node] = dis[current] + cost
                # 싸이클 유무 확인을 위해 n번 돌렸을 때
                # 최단 거리 갱신이 발생하면 음의 싸이클이 존재
                if i == n - 1:
                    return True
    return False


cycle = bf(1)
if cycle == True:    # 만약 음의 사이클 발생 시
    print(-1)
else:                # 만약 양의 사이클 발생 시
    for i in range(2, n + 1):
        if dis[i] == INF:
            print(-1)
        else:
            print(dis[i])

