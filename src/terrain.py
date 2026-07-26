"""산악 능선 지형 보정: 풍향 섹터별 NWP 바이어스 + 경험적 파워커브.

복잡지형에서 NWP 격자(1.5~25 km)는 능선의 가속·차폐를 해상하지 못하고,
그 오차는 풍향에 강하게 의존한다. 학습 기간 SCADA 실측 풍속으로
섹터별 승법 보정계수를 만들고, 보정 풍속을 SCADA 경험적 파워커브에 통과시켜
'물리적으로 타당한 출발점'을 피처로 제공한다 (2단계 파이프라인이 아니라 피처).

Data Leakage 준수:
- 보정표·파워커브는 **학습 연도 SCADA로만** 적합한다 (fit_years 인자로 강제).
  각 폴드는 자기 학습 연도만 쓰고, 최종 모델은 2022~2024만 쓴다.
- 2025 평가 구간에는 SCADA가 제공되지 않으며 사용하지도 않는다.
- 보정표는 학습으로 얻은 모델 파라미터와 동일한 지위이며,
  예측 시점에는 이미 확정된 과거 정보만 담는다.
"""
import numpy as np
import pandas as pd

from config import CAPACITY_KWH, GROUPS
from scada import _GROUP_TURBINES, _load_scada

N_SECTOR = 12          # 30도 폭 — 섹터당 표본 수와 해상도의 절충
WS_BINS = np.arange(0.0, 30.5, 0.5)
MIN_ROWS = 200


def scada_hourly_wind(index: pd.Index) -> pd.DataFrame:
    """그룹별 시간평균 실측 풍속·풍향(SCADA). 학습 기간에만 존재."""
    sc = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        d = sc[maker]
        cols_ws = [f"{maker}_wtg{t:02d}_ws" for t in turbines]
        cols_wd = [f"{maker}_wtg{t:02d}_wd" for t in turbines]
        ws = d[cols_ws].where(lambda x: (x >= 0) & (x <= 60))
        wd = np.deg2rad(d[cols_wd].to_numpy(dtype=float))
        hour = d.index.ceil("h")
        out[f"{g}_ws"] = ws.mean(axis=1).groupby(hour).mean().reindex(index)
        # 풍향은 벡터 평균 (원형 자료라 산술평균 불가)
        s = pd.Series(np.nanmean(np.sin(wd), axis=1), index=d.index).groupby(hour).mean()
        c = pd.Series(np.nanmean(np.cos(wd), axis=1), index=d.index).groupby(hour).mean()
        out[f"{g}_wd"] = np.degrees(np.arctan2(s, c)).reindex(index) % 360
    return pd.DataFrame(out, index=index)


def scada_hourly_power(index: pd.Index) -> pd.DataFrame:
    """그룹별 시간 출력(정격 대비) — 파워커브 적합용."""
    sc = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        d = sc[maker]
        pw = d[[f"{maker}_wtg{t:02d}_power_kw10m" for t in turbines]]
        pw = pw.where((pw >= -100) & (pw <= 800))
        hour = d.index.ceil("h")
        n = len(list(turbines))
        e = pw.sum(axis=1, min_count=n // 2).groupby(hour).sum(min_count=3)
        out[g] = (e / CAPACITY_KWH[g]).reindex(index)
    return pd.DataFrame(out, index=index)


def _sector(deg):
    return np.floor((np.asarray(deg, dtype=float) % 360) / (360.0 / N_SECTOR))


def fit_sector_bias(nwp_ws: pd.Series, nwp_dir_deg: pd.Series,
                    meas_ws: pd.Series, fit_index: pd.Index) -> np.ndarray:
    """섹터별 승법 보정계수 = median(실측 / NWP). 학습 연도 행만 사용."""
    x = nwp_ws.reindex(fit_index).to_numpy(dtype=float)
    y = meas_ws.reindex(fit_index).to_numpy(dtype=float)
    s = _sector(nwp_dir_deg.reindex(fit_index).to_numpy(dtype=float))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (x > 1.0) & (y > 1.0)
    x, y, s = x[ok], y[ok], s[ok]
    global_k = float(np.median(y / x)) if len(x) else 1.0
    k = np.full(N_SECTOR, global_k)
    for i in range(N_SECTOR):
        m = s == i
        if m.sum() >= MIN_ROWS:
            k[i] = float(np.median(y[m] / x[m]))
    # 표본이 적은 섹터가 튀지 않도록 물리적으로 타당한 범위로 제한
    return np.clip(k, 0.5, 2.0)


def apply_sector_bias(nwp_ws: pd.Series, nwp_dir_deg: pd.Series, k: np.ndarray) -> pd.Series:
    idx = _sector(nwp_dir_deg.to_numpy(dtype=float))
    idx = np.where(np.isfinite(idx), idx, 0).astype(int)
    return nwp_ws * k[idx]


def fit_power_curve(meas_ws: pd.Series, meas_power: pd.Series,
                    fit_index: pd.Index) -> np.ndarray:
    """풍속 0.5 m/s 빈별 출력 중앙값. 단조 증가로 정리해 뒤집힘을 막는다."""
    v = meas_ws.reindex(fit_index).to_numpy(dtype=float)
    p = meas_power.reindex(fit_index).to_numpy(dtype=float)
    ok = np.isfinite(v) & np.isfinite(p) & (p >= -0.05) & (p <= 1.2)
    v, p = v[ok], np.clip(p[ok], 0, 1)
    idx = np.clip(np.digitize(v, WS_BINS) - 1, 0, len(WS_BINS) - 2)
    curve = np.full(len(WS_BINS) - 1, np.nan)
    for i in range(len(curve)):
        m = idx == i
        if m.sum() >= 30:
            curve[i] = np.median(p[m])
    curve = pd.Series(curve).ffill().bfill().to_numpy()
    return np.maximum.accumulate(curve)


def apply_power_curve(ws, curve: np.ndarray) -> np.ndarray:
    v = np.asarray(ws, dtype=float)
    centers = WS_BINS[:-1] + 0.25
    out = np.interp(np.nan_to_num(v, nan=centers[0]), centers, curve)
    return np.where(np.isfinite(v), out, np.nan)


def _nwp_hub(feat):
    """NWP 허브풍속·풍향 — 두 예보원 평균 (문헌: 다중 NWP 결합이 단일보다 우수)."""
    ws = feat[["ldaps_ws117_mean", "gfs_ws117_mean"]].mean(axis=1)
    dr = np.degrees(np.arctan2(feat["gfs_dir_sin_mean"], feat["gfs_dir_cos_mean"])) % 360
    return ws, dr


def fit_terrain(train_feat: pd.DataFrame, fit_years) -> dict:
    """학습 기간 SCADA로 그룹별 (섹터 보정계수, 파워커브)를 적합한다.

    fit_years 밖의 행은 쓰지 않으므로, 검증 폴드는 자기 학습 연도만,
    최종 모델은 2022~2024만 보게 된다. 반환값은 학습으로 얻은 파라미터이며
    추론 시에는 이 표만 사용한다(2025 SCADA는 존재하지도, 사용하지도 않음).
    """
    wind = scada_hourly_wind(train_feat.index)
    power = scada_hourly_power(train_feat.index)
    fit_index = train_feat.index[train_feat.index.year.isin(list(fit_years))]
    nwp_ws, nwp_dir = _nwp_hub(train_feat)
    return {g: {"k": fit_sector_bias(nwp_ws, nwp_dir, wind[f"{g}_ws"], fit_index),
                "curve": fit_power_curve(wind[f"{g}_ws"], power[g], fit_index)}
            for g in GROUPS}


def apply_terrain(feat: pd.DataFrame, params: dict) -> pd.DataFrame:
    """적합된 보정표를 임의 구간(학습·검증·2025 평가)에 적용해 피처를 만든다."""
    nwp_ws, nwp_dir = _nwp_hub(feat)
    cols = {}
    for g in GROUPS:
        k, curve = params[g]["k"], params[g]["curve"]
        ws_c = apply_sector_bias(nwp_ws, nwp_dir, k)
        n = g[-1]
        cols[f"tc{n}_ws"] = ws_c
        cols[f"tc{n}_pc"] = apply_power_curve(ws_c, curve)
        cols[f"tc{n}_pc_raw"] = apply_power_curve(nwp_ws, curve)
        cols[f"tc{n}_k"] = ws_c / np.maximum(nwp_ws, 0.5)
    cols["t_sector"] = _sector(nwp_dir)
    return pd.DataFrame(cols, index=feat.index)


def build_terrain_features(feat: pd.DataFrame, fit_years) -> pd.DataFrame:
    """학습 구간 전용 편의 함수 (적합과 적용이 같은 프레임일 때)."""
    return apply_terrain(feat, fit_terrain(feat, fit_years))


def save_terrain(params: dict, path):
    np.savez(path, **{f"{g}|{k}": v for g, d in params.items() for k, v in d.items()})


def load_terrain(path) -> dict:
    z = np.load(path)
    return {g: {"k": z[f"{g}|k"], "curve": z[f"{g}|curve"]} for g in GROUPS}


def demo():
    """자체 점검: 보정표가 학습 연도만 보는지, 파워커브가 단조인지."""
    idx = pd.date_range("2022-01-01 01:00", "2024-12-31 23:00", freq="h")
    rng = np.random.default_rng(0)
    ws = pd.Series(rng.gamma(4, 1.5, len(idx)), index=idx)
    dr = pd.Series(rng.uniform(0, 360, len(idx)), index=idx)
    meas = ws * 1.3                      # 균일한 30% 과소예보
    fit_idx = idx[idx.year == 2022]
    k = fit_sector_bias(ws, dr, meas, fit_idx)
    assert np.allclose(k, 1.3, atol=0.05), k
    curve = fit_power_curve(meas, pd.Series(np.clip(meas / 15, 0, 1), index=idx), fit_idx)
    assert np.all(np.diff(curve) >= -1e-12), "파워커브가 단조가 아님"
    assert apply_power_curve([0.1], curve)[0] <= apply_power_curve([20.0], curve)[0]
    print("terrain demo 통과 ✓")


if __name__ == "__main__":
    demo()
