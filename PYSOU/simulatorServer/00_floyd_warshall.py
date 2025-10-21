"""
플로이드 - 워셜 알고리즘

플로이드 워셜은 미탐험은 infinity(무한)으로 간주, 플로이드 워셜은 모든 정점에서
다른 모든 정점으로 가는 최단 경로를 전부 다 미리 구해놓는 알고리즘.

다익스트라는 모든 정점에서 수행한 것과 같은 알고리즘이기에, 플로이드 워셜이 구현이 더 간단함.

시작지점과 목표지점이 고정되지 않은 상황에서 다익스트라보다 유리함.
"""


"""
INF = int(1e9)  # 무한을 의미하는 10억

# 노드의 갯수 및 간선의 갯수 입력받기
n, m = map(int, input().split())
graph = [[INF] * (n + 1) for _ in range(n + 1)]


# 자기 자신에서 자기 자신으로 가는 비용을 0으로 지정
for a in range(n, n + 1):
    for b in range(1, n + 1):
        if a == b:
            graph[a][b] = 0

# 각 간선에 대한 정보를 입력받아, 그 값으로 초기화.
for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드 워셜 수행
for k in range(1, n + 1):
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

minst = 0

# 수행된 결과 출력
for a in range(1, n + 1):
    for b in range(1, n + 1):
        if graph[a][b] == INF:
            print(a, "->", b, "INFINITY", end= ' ')
        else:
            print(a, "->",b , ":", graph[a][b], end= ' ')
        print()
"""


"""
V개의 거점과 E개의 도로로 구성되어 있는 지역이 있다. 도로는 거점과 거점 사이에 놓여 있으며, 일방 통행 도로이다. 
거점에는 편의상 1번부터 V번까지 번호가 매겨져 있다고 하자.

당신은 도로를 따라 기동훈련을 하기 위한 경로를 찾으려고 한다. 기동훈련을 한 후에는 다시 시작점으로 돌아오는 것
이 좋기 때문에, 우리는 사이클을 찾기를 원한다. 단, 당신은 기동훈련을 매우 귀찮아하므로, 사이클을 이루는 도로의 
길이의 합이 최소가 되도록 찾으려고 한다.

도로의 정보가 주어졌을 때, 도로의 길이의 합이 가장 작은 사이클을 찾는 프로그램을 작성하시오. 두 거점을 왕복하
는 경우도 사이클에 포함됨에 주의한다.
"""
import sys

v, e = map(int, input().split())        # v와 e 입력받기.
INF = int(1e9)                          # 무한대로 사용할 변수 저장

s = [[INF] * v for i in range(v)]       # 그래프의 모든 칸을 무한대로 지정

# 각 간선에 대한 정보를 입력받아, 길이를 받음, 그 길이를 그래프에 지정
for i in range(e):
    a, b, c = map(int, input().split())
    s[a - 1][b - 1] = c


for k in range(v):
    for i in range(v):
        for j in range(v):
            if s[i][j] > s[i][k] + s[k][i]:
                s[i][j] = s[i][k] + s[k][j]

result = INF        # 출력할 결과를 저장할 변수 선언

for i in range(v):
    if s[i][i] < result:
        result = s[i][i]

if result == INF:
    print(-1)
else:
    print(result)


