"""터빈 좌표 파싱 및 기상 격자 거리가중(IDW) 계산."""
import re

import numpy as np
import pandas as pd

from config import DATA_DIR, GROUPS

_DMS_RE = re.compile(r"""(\d+)°(\d+)'([\d.]+)"([NSEW])""")


def _dms_to_decimal(token: str) -> float:
    deg, minute, sec, hemi = _DMS_RE.match(token).groups()
    val = float(deg) + float(minute) / 60 + float(sec) / 3600
    return -val if hemi in ("S", "W") else val


def load_turbine_coords() -> pd.DataFrame:
    """info.xlsx에서 터빈별 (그룹, 위도, 경도) 추출.

    VESTAS V126 1~6호기=그룹1, 7~12호기=그룹2, UNISON U136 1~5호기=그룹3.
    """
    raw = pd.read_excel(DATA_DIR / "info.xlsx", header=None)
    header_row = raw.index[raw.iloc[:, 1].eq("단계")][0]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row]
    df = df.dropna(subset=["좌표(Google)"])

    rows = []
    for _, r in df.iterrows():
        lat_tok, lon_tok = str(r["좌표(Google)"]).split()
        maker, unit = str(r["제작사"]).strip(), int(r["호기"])
        if maker == "VESTAS":
            group = "kpx_group_1" if unit <= 6 else "kpx_group_2"
        else:
            group = "kpx_group_3"
        rows.append({"group": group, "lat": _dms_to_decimal(lat_tok), "lon": _dms_to_decimal(lon_tok)})
    return pd.DataFrame(rows)


def group_centroids() -> dict:
    coords = load_turbine_coords()
    return {g: (sub["lat"].mean(), sub["lon"].mean())
            for g, sub in coords.groupby("group")}


def idw_weights(grid_latlon: pd.DataFrame, power: float = 2.0) -> dict:
    """격자(grid_id, latitude, longitude) → 그룹별 정규화 IDW 가중치 Series."""
    weights = {}
    for g in GROUPS:
        clat, clon = group_centroids()[g]
        # 위도/경도 → km 근사 (해당 위도에서 경도 1도 ≈ 88.5km)
        dlat = (grid_latlon["latitude"] - clat) * 111.0
        dlon = (grid_latlon["longitude"] - clon) * 88.5
        dist = np.sqrt(dlat**2 + dlon**2).clip(lower=0.1)
        w = 1.0 / dist**power
        weights[g] = pd.Series((w / w.sum()).to_numpy(), index=grid_latlon["grid_id"].to_numpy())
    return weights
