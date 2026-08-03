"""GEFS 피처 생성 — 취득한 상층풍을 학습용 피처로 변환.

표본 검정에서 GEFS 850 hPa 바람이 제공 LDAPS(1.5 km)보다 SCADA 실측 허브풍속을
더 잘 설명했다(상관 0.83~0.85 vs 0.71~0.78). 사이트가 해발 약 1,000 m 산악이라
허브(지상 117 m)가 대략 890 hPa 에 놓이기 때문이다.

**시간 보간의 누수 검토**: GEFS 0.5도는 3시간 간격이라 시간 보간이 필요하다.
각 발표분(00Z 런)은 KST 익일 03:00~익익일 00:00 을 덮으므로, 대상 시각 01:00·02:00 는
직전 발표분(하루 전 13:00 공개)의 끝값과 이어 보간된다. 두 발표분 모두 예측기준시점
(전일 13:00 KST) 이전에 공개된 것이라 규칙을 위반하지 않는다.
"""
import numpy as np
import pandas as pd

from config import ROOT

CACHE = ROOT / "cache"


def _ws(u, v):
    return np.sqrt(u**2 + v**2)


def build(index: pd.DatetimeIndex) -> pd.DataFrame:
    """주어진 시각 인덱스에 맞춘 GEFS 파생 피처."""
    g = pd.read_parquet(CACHE / "gefs_hourly.parquet").reindex(index)
    f = pd.DataFrame(index=index)

    for tag in ("c", "mean"):          # c = 사이트 최근접 격자, mean = 3x3 평균
        u9, v9 = g[f"gefs_u925_{tag}"], g[f"gefs_v925_{tag}"]
        u8, v8 = g[f"gefs_u850_{tag}"], g[f"gefs_v850_{tag}"]
        ws9, ws8 = _ws(u9, v9), _ws(u8, v8)
        f[f"ge_ws925_{tag}"] = ws9
        f[f"ge_ws850_{tag}"] = ws8
        # 허브(약 890 hPa)는 925 와 850 사이 — 기압 선형 보간 가중치
        w = (925.0 - 890.0) / (925.0 - 850.0)
        f[f"ge_wshub_{tag}"] = (1 - w) * ws9 + w * ws8
        f[f"ge_wshub_cube_{tag}"] = f[f"ge_wshub_{tag}"] ** 3
        # 연직 시어 — 두 고도 간 풍속·풍향 변화
        f[f"ge_shear_{tag}"] = ws8 - ws9
        n = np.sqrt((u9**2 + v9**2) * (u8**2 + v8**2)) + 1e-6
        f[f"ge_veer_cos_{tag}"] = (u9 * u8 + v9 * v8) / n
        f[f"ge_veer_sin_{tag}"] = (u9 * v8 - v9 * u8) / n
        for lab, u, v in (("925", u9, v9), ("850", u8, v8)):
            th = np.arctan2(v, u)
            f[f"ge_dir{lab}_sin_{tag}"] = np.sin(th)
            f[f"ge_dir{lab}_cos_{tag}"] = np.cos(th)

    # 격자 간 편차 = 공간 경도(전선 접근 등의 신호)
    f["ge_grad_ws850"] = f["ge_ws850_c"] - f["ge_ws850_mean"]
    f["ge_grad_ws925"] = f["ge_ws925_c"] - f["ge_ws925_mean"]
    # 같은 범위에 딸려 온 기온·습도·지위고도
    for c in ("gefs_t_c", "gefs_r_c", "gefs_gh_c", "gefs_t_mean", "gefs_gh_mean"):
        if c in g.columns:
            f[f"ge_{c[5:]}"] = g[c]
    return f


def add_cross(feat: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """제공 예보와 GEFS 의 불일치도 — 다중 NWP 결합의 핵심 신호."""
    out = gf.copy()
    for col, ref in (("ldaps_ws117_mean", "ge_wshub_c"), ("gfs_ws117_mean", "ge_wshub_c")):
        if col in feat.columns:
            out[f"x_{col[:5]}_ge_diff"] = feat[col] - gf[ref]
            out[f"x_{col[:5]}_ge_mean"] = (feat[col] + gf[ref]) / 2
    if {"ldaps_ws117_mean", "gfs_ws117_mean"}.issubset(feat.columns):
        tri = np.c_[feat["ldaps_ws117_mean"], feat["gfs_ws117_mean"], gf["ge_wshub_c"]]
        out["x_tri_mean"] = tri.mean(axis=1)
        out["x_tri_std"] = tri.std(axis=1)      # 3모델 불일치 = 예보 불확실성
    return out


def demo():
    """자체 점검: 인덱스 정렬과 결측률."""
    tr = pd.read_parquet(CACHE / "full_train.parquet")
    gf = build(tr.index)
    miss = gf.isna().mean().mean()
    assert len(gf) == len(tr), "인덱스 길이 불일치"
    assert miss < 0.05, f"결측률 과다: {miss:.2%}"
    x = add_cross(tr, gf)
    assert "x_tri_std" in x.columns and x["x_tri_std"].notna().mean() > 0.9
    print(f"gefs_feat demo 통과 ✓  {gf.shape[1]}피처(+교차 {x.shape[1]-gf.shape[1]}), 결측 {miss:.2%}")


if __name__ == "__main__":
    demo()
