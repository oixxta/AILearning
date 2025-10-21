import heapq

def manhattan(a, b):
    """4방향 격자에서 쓰는 대표 휴리스틱(과대평가 X → 최단경로 보장)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors_4dir(r, c):
    """상하좌우 이웃 좌표 생성."""
    return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

def reconstruct_path(came_from, current):
    """came_from를 따라가서 시작→도착 순서로 경로를 돌려준다."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def a_star_4dir(grid, start, goal):
    """
    grid: 0은 통과 가능, 1은 벽(장애물)
    start/goal: (row, col)
    반환: 경로(리스트[(r,c), ...]) 또는 None
    """
    rows, cols = len(grid), len(grid[0])

    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def walkable(r, c):  return in_bounds(r, c) and grid[r][c] == 0

    if not walkable(*start) or not walkable(*goal):
        return None

    # 우선순위 큐(open): (f, g, (r,c))
    g_score = {start: 0}
    h_start = manhattan(start, goal)
    open_heap = [(h_start, 0, start)]
    came_from = {}
    closed = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:           # 이미 처리한 노드면 패스
            continue
        closed.add(current)

        if current == goal:             # 목표에 도달 → 경로 복원
            return reconstruct_path(came_from, current)

        cr, cc = current
        for nr, nc in neighbors_4dir(cr, cc):
            if not walkable(nr, nc):
                continue
            nxt = (nr, nc)

            tentative_g = g_score[current] + 1  # 4방향은 이동비용 1
            if nxt in closed and tentative_g >= g_score.get(nxt, float("inf")):
                continue

            # 더 좋은 경로를 찾았으면 갱신
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_new = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, nxt))

    return None  # 경로가 없을 때

# ====== 데모 실행 ======
if __name__ == "__main__":
    # 0 = 빈 칸, 1 = 벽
    grid = [
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    start = (0, 0)
    goal  = (7, 6)

    path = a_star_4dir(grid, start, goal)
    print("경로:", path)

    # 보기 좋게 표시
    if path:
        view = [row[:] for row in grid]           # 깊은 복사
        for r, c in path:
            view[r][c] = '*'
        sr, sc = start
        gr, gc = goal
        view[sr][sc] = 'S'
        view[gr][gc] = 'G'

        print("\n격자( S=start, G=goal, *=path, 1=벽 )")
        for row in view:
            print(" ".join(str(x) for x in row))
    else:
        print("경로가 없습니다.")