import pandas as pd
import numpy as np
from datetime import datetime

def build_full_grid(max_coord=300):
    """
    0 ~ max_coord-1 범위의 X,Z 정수 격자를 모두 생성 (총 max_coord^2 행)
    ※ 지도명이 300*300이면 좌표는 [0, 299]가 자연스럽습니다.
    """
    xs = np.arange(0, max_coord, dtype=np.int32)
    zs = np.arange(0, max_coord, dtype=np.int32)
    grid = pd.MultiIndex.from_product([xs, zs], names=["Player_Pos_X", "Player_Pos_Z"]).to_frame(index=False)
    return grid

def idw_fill(
    known_x, known_z, known_y,
    query_x, query_z,
    k=8, power=2.0, chunk=4000, eps=1e-6
):
    """
    순수 NumPy로 구현한 IDW 보간.
    - (query_x, query_z): 결측 Y를 채울 좌표 배열
    - (known_x, known_z, known_y): 관측(기존) 샘플
    - k: 최근접 이웃 개수
    - power: 거리 가중 지수 (2 권장)
    - chunk: 메모리 절약용 청크 크기 (결측 수가 많아도 안전)
    반환: 보간된 Y(np.ndarray, float)
    """
    known = np.stack([known_x, known_z], axis=1).astype(np.float32)
    query = np.stack([query_x, query_z], axis=1).astype(np.float32)

    out = np.empty(query.shape[0], dtype=np.float32)

    # 청크 단위로 거리 계산 → k개 이웃 선택 → 가중 평균
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        q = query[start:end]  # (C, 2)

        # 모든 known과의 거리 계산 (C, N)
        # d^2 = (dx^2 + dz^2)
        dx = q[:, None, 0] - known[None, :, 0]
        dz = q[:, None, 1] - known[None, :, 1]
        dist2 = dx*dx + dz*dz

        # 자기 자신과 겹치는 경우(거리 0) → 해당 y 그대로 할당
        zero_mask = (dist2 <= eps)
        any_zero = zero_mask.any(axis=1)
        if any_zero.any():
            # 행마다 첫 true의 인덱스 찾아 그 known_y 그대로 써줌
            idx_zero = zero_mask.argmax(axis=1)
            exact_vals = known_y[idx_zero]
        else:
            exact_vals = None

        # k개 최근접 이웃 인덱스
        # np.argpartition: O(N)으로 상위/하위 k 뽑기
        nn_idx = np.argpartition(dist2, kth=min(k, dist2.shape[1]-1), axis=1)[:, :k]  # (C, k)
        # 선택된 이웃들의 실제 거리
        dsel2 = np.take_along_axis(dist2, nn_idx, axis=1) + eps  # 0 div 방지 eps
        w = 1.0 / (dsel2 ** (power/2.0))  # (C, k) ; power=2 -> 1/d^1

        ysel = known_y[nn_idx]  # (C, k)
        y_idw = (w * ysel).sum(axis=1) / w.sum(axis=1)  # (C,)

        # 정확히 겹친 점은 exact_vals 사용
        if any_zero.any():
            y_idw[any_zero] = exact_vals[any_zero]

        out[start:end] = y_idw.astype(np.float32)

    return out

def fill_missing_altitude_to_full_grid(
    csv_path: str,
    map_size: int = 300,
    method: str = "idw",
    k: int = 8,
    power: float = 2.0,
    save: bool = True,
    out_name_prefix: str = "altatude_map"
):
    """
    - 입력: Player_Pos_X, Player_Pos_Y, Player_Pos_Z (정수화 및 중복제거된 CSV)
    - 출력: 300x300 전체 격자(0~299)를 모두 갖고, Y(고도) 결측을 보간한 DataFrame
    - method: "idw" | "nearest"
    """
    raw = pd.read_csv(csv_path)
    # 안전장치: 필요한 컬럼만, 범위내 정수만
    cols = ["Player_Pos_X", "Player_Pos_Y", "Player_Pos_Z"]
    if not set(cols).issubset(raw.columns):
        raise ValueError(f"CSV에 {cols} 컬럼이 필요합니다.")

    df = raw[cols].copy()
    # 정수화(소수점 이하 버림) 및 좌표 범위 클리핑
    df["Player_Pos_X"] = df["Player_Pos_X"].astype(np.int32).clip(0, map_size-1)
    df["Player_Pos_Z"] = df["Player_Pos_Z"].astype(np.int32).clip(0, map_size-1)
    # 같은 (X,Z)면 첫 행만 유지
    df = df.drop_duplicates(subset=["Player_Pos_X", "Player_Pos_Z"], keep="first").reset_index(drop=True)

    # 전체 격자 생성 후 병합
    full_grid = build_full_grid(map_size)
    merged = full_grid.merge(df, on=["Player_Pos_X", "Player_Pos_Z"], how="left", suffixes=("", "_orig"))
    # 이제 merged에는 모든 (X,Z)가 존재, Player_Pos_Y가 NaN인 셀들이 결측.

    # 관측치/결측치 분리
    known_mask = ~merged["Player_Pos_Y"].isna()
    miss_mask = ~known_mask

    if miss_mask.any():
        known_x = merged.loc[known_mask, "Player_Pos_X"].to_numpy(np.int32)
        known_z = merged.loc[known_mask, "Player_Pos_Z"].to_numpy(np.int32)
        known_y = merged.loc[known_mask, "Player_Pos_Y"].to_numpy(np.float32)

        query_x = merged.loc[miss_mask, "Player_Pos_X"].to_numpy(np.int32)
        query_z = merged.loc[miss_mask, "Player_Pos_Z"].to_numpy(np.int32)

        if method == "idw":
            filled_y = idw_fill(
                known_x, known_z, known_y,
                query_x, query_z,
                k=k, power=power, chunk=4000
            )
        elif method == "nearest":
            # k=1, power=2 의 특수 케이스(최근접 이웃 값 복사)
            filled_y = idw_fill(
                known_x, known_z, known_y,
                query_x, query_z,
                k=1, power=2.0, chunk=4000
            )
        else:
            raise ValueError("method는 'idw' 또는 'nearest'만 지원합니다.")

        merged.loc[miss_mask, "Player_Pos_Y"] = filled_y

    # 열 순서 통일
    merged = merged[["Player_Pos_X", "Player_Pos_Y", "Player_Pos_Z"]]

    # 저장 (파일명: [map name]altatude_map_YYYYMMDD_HHMMSS.csv)
    out_path = None
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{out_name_prefix}_{ts}.csv"
        merged.to_csv(out_path, index=False)
        print(f"[메서드4] 결측 보간된 altatude_map 저장 완료 → {out_path}")

    return merged, out_path