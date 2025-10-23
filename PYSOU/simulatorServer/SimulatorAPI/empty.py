import math

def generate_circle_points(x, y, r, num_points=8):
    """
    중심 (x, y)와 반지름 r을 기준으로
    원 둘레 위에 균등하게 배치된 num_points개의 좌표를 반환합니다.
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points  # 각도 (라디안 단위)
        px = x + r * math.cos(angle)
        py = y + r * math.sin(angle)
        points.append((px, py))
    return points


# 예시
center_x, center_y, radius = 10, 5, 3
points = generate_circle_points(center_x, center_y, radius)

print(len(points))
print(points[0])