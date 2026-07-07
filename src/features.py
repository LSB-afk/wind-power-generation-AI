"""LDAPS/GFS 기상예보 데이터 피처 엔지니어링 (train/test 공용).

제공된 예보 데이터는 전일 13:00(KST) 발표분만 포함되어 있으므로
그대로 사용해도 예측기준시점(Data Leakage) 규칙을 준수한다.
"""
import numpy as np
import pandas as pd


def _wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


def _wind_dir_sincos(u, v, prefix):
    """풍향(기상학적 방위 무관, 모델용 sin/cos 표현)."""
    theta = np.arctan2(v, u)
    return {f"{prefix}_wd_sin": np.sin(theta), f"{prefix}_wd_cos": np.cos(theta)}


def build_ldaps_features(path):
    """LDAPS(1.5km, 16격자): 격자별 풍속 + 전체 격자 집계 피처.

    반환: (피처 DataFrame, 예보 대상 시각별 data_available 시각 Series)
    """
    df = pd.read_csv(path, encoding="utf-8-sig",
                     parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"])

    # 격자별(행 단위) 파생: 풍속
    df["ws10"] = _wind_speed(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
    df["ws50max"] = _wind_speed(df["heightAboveGround_50_50MUmax"], df["heightAboveGround_50_50MVmax"])
    df["ws50min"] = _wind_speed(df["heightAboveGround_50_50MUmin"], df["heightAboveGround_50_50MVmin"])
    df["blws"] = _wind_speed(df["heightAboveGround_5_XBLWS"], df["heightAboveGround_5_YBLWS"])
    for k, v in _wind_dir_sincos(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"], "d10").items():
        df[k] = v
    # 공기밀도 근사 rho = P / (R * T)
    df["air_density"] = df["surface_0_sp"] / (287.05 * df["heightAboveGround_2_t"])

    # 격자별 개별 피처 (바람 관련만 — 격자 간 지형 차이가 큼)
    grid_vars = ["ws10", "ws50max", "ws50min", "blws", "d10_wd_sin", "d10_wd_cos"]
    wide = df.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=grid_vars)
    wide.columns = [f"ldaps_g{int(g):02d}_{v}" for v, g in wide.columns]

    # 전체 격자 집계 피처
    agg_vars = {
        "ws10": ["mean", "std", "max"],
        "ws50max": ["mean", "max"],
        "blws": ["mean"],
        "heightAboveGround_2_t": ["mean"],
        "heightAboveGround_2_r": ["mean"],
        "surface_0_sp": ["mean"],
        "air_density": ["mean"],
        "etc_0_blh": ["mean"],
        "etc_0_lcc": ["mean"],
        "surface_0_avg_lsprate": ["mean"],
        "surface_0_ncpcp": ["mean"],
    }
    agg = df.groupby("forecast_kst_dtm").agg(agg_vars)
    agg.columns = [f"ldaps_{c}_{s}" for c, s in agg.columns]

    davail = df.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    return wide.join(agg), davail


def build_gfs_features(path) -> pd.DataFrame:
    """GFS(0.25도, 9격자): 허브고도(117m) 인접 80/100m 풍속 중심."""
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])

    df["ws10"] = _wind_speed(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
    df["ws80"] = _wind_speed(df["heightAboveGround_80_u"], df["heightAboveGround_80_v"])
    df["ws100"] = _wind_speed(df["heightAboveGround_100_100u"], df["heightAboveGround_100_100v"])
    df["ws850"] = _wind_speed(df["isobaricInhPa_850_u"], df["isobaricInhPa_850_v"])
    df["ws_pbl"] = _wind_speed(df["planetaryBoundaryLayer_0_u"], df["planetaryBoundaryLayer_0_v"])
    # 연직 시어(고도별 풍속 차) — 산악 지형 난류 특성
    df["shear_100_10"] = df["ws100"] - df["ws10"]
    for k, v in _wind_dir_sincos(df["heightAboveGround_100_100u"], df["heightAboveGround_100_100v"], "d100").items():
        df[k] = v
    df["air_density"] = df["surface_0_sp"] / (287.05 * df["heightAboveGround_2_2t"])
    # 풍력 에너지는 풍속의 세제곱에 비례
    df["ws100_cube"] = df["ws100"] ** 3

    grid_vars = ["ws10", "ws80", "ws100", "ws100_cube", "surface_0_gust", "d100_wd_sin", "d100_wd_cos"]
    wide = df.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=grid_vars)
    wide.columns = [f"gfs_g{int(g)}_{v}" for v, g in wide.columns]

    agg_vars = {
        "ws10": ["mean"],
        "ws80": ["mean", "std"],
        "ws100": ["mean", "std", "max", "min"],
        "ws100_cube": ["mean"],
        "ws850": ["mean"],
        "ws_pbl": ["mean"],
        "shear_100_10": ["mean"],
        "surface_0_gust": ["mean", "max"],
        "heightAboveGround_2_2t": ["mean"],
        "air_density": ["mean"],
        "surface_0_prate": ["mean"],
        "atmosphere_0_tcc": ["mean"],
    }
    agg = df.groupby("forecast_kst_dtm").agg(agg_vars)
    agg.columns = [f"gfs_{c}_{s}" for c, s in agg.columns]

    return wide.join(agg)


def add_time_features(feat: pd.DataFrame) -> pd.DataFrame:
    """시간 주기성 피처 (예보 대상 시각 기준 — 누수 없음)."""
    idx = feat.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    feat["month"] = idx.month
    return feat


# 컨텍스트(발표분 내 시계열) 피처 대상 컬럼
CONTEXT_COLS = [
    "ldaps_ws10_mean", "ldaps_ws10_max", "ldaps_ws50max_mean", "ldaps_blws_mean",
    "gfs_ws100_mean", "gfs_ws80_mean", "gfs_ws100_cube_mean",
    "gfs_surface_0_gust_mean", "gfs_ws_pbl_mean",
]


def add_context(feat: pd.DataFrame, davail: pd.Series) -> pd.DataFrame:
    """같은 발표분(data_available) 24시간 블록 내 lag/lead/rolling.

    블록 경계는 NaN — 다른 발표분(미래 공개 예보) 정보가 섞이지 않아 누수가 없다.
    """
    feat = feat.copy()
    block = davail.reindex(feat.index)
    new_cols = {}
    for c in CONTEXT_COLS:
        if c not in feat.columns:
            continue
        g = feat[c].groupby(block)
        for s in (-2, -1, 1, 2):
            new_cols[f"{c}_sh{s}"] = g.shift(s)
        for w in (3, 6):
            roll = g.rolling(w, center=True, min_periods=2)
            new_cols[f"{c}_rm{w}"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{c}_rs{w}"] = roll.std().reset_index(level=0, drop=True)
        new_cols[f"{c}_d1"] = feat[c] - g.shift(1)
    return pd.concat([feat, pd.DataFrame(new_cols, index=feat.index)], axis=1)


def build_features(ldaps_path, gfs_path, context: bool = True) -> pd.DataFrame:
    """예보 대상 시각(forecast_kst_dtm) 인덱스의 피처 테이블 생성."""
    ldaps, davail = build_ldaps_features(ldaps_path)
    gfs = build_gfs_features(gfs_path)
    feat = ldaps.join(gfs, how="outer")
    feat = add_time_features(feat)
    if context:
        feat = add_context(feat, davail)
    return feat.sort_index()
