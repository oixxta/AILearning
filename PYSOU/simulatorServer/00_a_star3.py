import heapq
import time
import argparse
from typing import List, Tuple, Optional

Coord = Tuple[int, int]
Grid = List[List[int]]

def manhattan(a: Coord, b: Coord) -> int:
    """4방향 격자에서 쓰는 대표 휴리스틱(과대평가 X → 최단경로 보장)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors_4dir(r: int, c: int):
    """상하좌우 이웃 좌표 생성."""
    return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

def reconstruct_path(came_from, current: Coord):
    """came_from를 따라가서 시작→도착 순서로 경로를 돌려준다."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def a_star_4dir(grid: Grid, start: Coord, goal: Coord):
    """
    grid: 0은 통과 가능, 1은 벽(장애물)
    start/goal: (row, col)
    반환: (경로 리스트 또는 None, 확장 노드 수)
    """
    rows, cols = len(grid), len(grid[0])

    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def walkable(r, c):  return in_bounds(r, c) and grid[r][c] == 0

    if not walkable(*start) or not walkable(*goal):
        return None, 0

    g_score = {start: 0}
    open_heap = [(manhattan(start, goal), 0, start)]  # (f, g, node)
    came_from = {}
    closed = set()
    expansions = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            return reconstruct_path(came_from, current), expansions

        cr, cc = current
        for nr, nc in neighbors_4dir(cr, cc):
            if not walkable(nr, nc):
                continue
            nxt = (nr, nc)
            tentative_g = g_score[current] + 1  # 4방향은 이동비용 1

            if nxt in closed and tentative_g >= g_score.get(nxt, float("inf")):
                continue

            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_new = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f_new, tentative_g, nxt))

    return None, expansions  # 경로가 없을 때

def tile_expand(base: Grid, r_mult: int, c_mult: int) -> Grid:
    """base 격자를 r_mult×c_mult로 타일처럼 복제하여 확장."""
    br, bc = len(base), len(base[0])
    out = [[0]*(bc*c_mult) for _ in range(br*r_mult)]
    for i in range(br*r_mult):
        for j in range(bc*c_mult):
            out[i][j] = base[i % br][j % bc]
    return out

def find_fallback_goal(grid: Grid, preferred: Coord) -> Optional[Coord]:
    """선호 goal이 벽이면 오른쪽 아래에서부터 가장 가까운 통로(0)를 찾아 반환."""
    r, c = preferred
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0:
        return preferred
    # 우하단부터 역순 스캔
    for i in range(len(grid)-1, -1, -1):
        for j in range(len(grid[0])-1, -1, -1):
            if grid[i][j] == 0:
                return (i, j)
    return None

def pretty_print(grid: Grid, path: Optional[List[Coord]], start: Coord, goal: Coord):
    if not path:
        print("경로가 없습니다.")
        return
    view = [row[:] for row in grid]
    for r, c in path:
        view[r][c] = '*'
    sr, sc = start
    gr, gc = goal
    view[sr][sc] = 'S'
    view[gr][gc] = 'G'
    print("\n격자( S=start, G=goal, *=path, 1=벽 )")
    for row in view:
        print(" ".join(str(x) for x in row))

def main():
    parser = argparse.ArgumentParser(description="A* 4방향 타일 확장 + 시간 측정")
    parser.add_argument("r_mult", nargs="?", type=int, default=1, help="세로(행) 타일 확장 배수 (기본=1)")
    parser.add_argument("c_mult", nargs="?", type=int, default=1, help="가로(열) 타일 확장 배수 (기본=1)")
    parser.add_argument("--no-print", action="store_true", help="격자와 경로를 출력하지 않음")
    args = parser.parse_args()

    # 원본 격자 (0 = 빈 칸, 1 = 벽)
    base_grid = [
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

    # 타일 확장
    r_mult, c_mult = max(1, args.r_mult), max(1, args.c_mult)
    grid = tile_expand(base_grid, r_mult, c_mult)
    rows, cols = len(grid), len(grid[0])

    # goal은 우하단을 선호(벽이면 우하단부터 가장 가까운 통로로 대체)
    preferred_goal = (rows - 1, cols - 1)
    goal = find_fallback_goal(grid, preferred_goal)
    if goal is None:
        print("통과 가능한 칸(0)이 전혀 없습니다. 종료합니다.")
        return

    #print(f"격자 크기: {rows} x {cols} (확장 배수: {r_mult} x {c_mult})")
    #print(f"start={start}, goal={goal}")

    # 시간 측정
    t0 = time.perf_counter()
    path, expansions = a_star_4dir(grid, start, goal)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    if path:
        print(f"경로 길이: {len(path)}")
    else:
        print("경로 없음")
    print(f"A* 연산 시간: {elapsed_ms:.3f} ms")
    print(f"확장(처리)한 노드 수: {expansions}")

    if not args.no_print:
        pretty_print(grid, path, start, goal)

if __name__ == "__main__":
    main()