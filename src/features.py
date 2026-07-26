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


def _veer(u1, v1, u2, v2, prefix):
    """두 고도 풍향 사이각의 sin/cos (풍향 전단 = veer)."""
    n = np.sqrt((u1**2 + v1**2) * (u2**2 + v2**2)) + 1e-6
    return {f"{prefix}_cos": (u1 * u2 + v1 * v2) / n,
            f"{prefix}_sin": (u1 * v2 - v1 * u2) / n}


def build_phys_features(ldaps_path, gfs_path) -> pd.DataFrame:
    """허브고도(117 m) 물리 피처 팩 — 전단지수·안정도·상층풍·환기율.

    기존 피처가 쓰지 않던 원본 변수(700/500 hPa 바람, 850 hPa 기온·습도,
    VRATE, 이슬점, 상층운)를 사용한다. 풍속의 고도 외삽은 멱법칙
    `ws(z) = ws100 * (z/100)^alpha`, `alpha = ln(ws100/ws10)/ln(10)` 로,
    대기 안정도가 이 alpha를 좌우하므로 안정도 지표를 함께 넣는다.
    모든 컬럼은 `px_` 접두사를 쓴다 (기존 피처명과 충돌 없음).
    """
    eps = 0.5
    g = pd.read_csv(gfs_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    ws10 = _wind_speed(g["heightAboveGround_10_10u"], g["heightAboveGround_10_10v"])
    ws100 = _wind_speed(g["heightAboveGround_100_100u"], g["heightAboveGround_100_100v"])
    gd = pd.DataFrame({"forecast_kst_dtm": g["forecast_kst_dtm"]})
    gd["px_alpha"] = np.log(np.maximum(ws100, eps) / np.maximum(ws10, eps)) / np.log(10.0)
    gd["px_ws117"] = ws100 * (1.17 ** gd["px_alpha"])
    gd["px_ws117_cube"] = gd["px_ws117"] ** 3
    rho = g["surface_0_sp"] / (287.05 * g["heightAboveGround_2_2t"])
    gd["px_pdens117"] = 0.5 * rho * gd["px_ws117_cube"]
    gd["px_ws700"] = _wind_speed(g["isobaricInhPa_700_u"], g["isobaricInhPa_700_v"])
    gd["px_ws500"] = _wind_speed(g["isobaricInhPa_500_u"], g["isobaricInhPa_500_v"])
    gd["px_ws850_ratio"] = _wind_speed(g["isobaricInhPa_850_u"],
                                       g["isobaricInhPa_850_v"]) / np.maximum(ws100, eps)
    # 안정도: 하층 기온감률이 클수록 불안정 → 연직 혼합 강화 → 전단 감소
    gd["px_lapse_2_850"] = g["heightAboveGround_2_2t"] - g["isobaricInhPa_850_t"]
    gd["px_lapse_850_700"] = g["isobaricInhPa_850_t"] - g["isobaricInhPa_700_t"]
    gd["px_dpt_dep"] = g["heightAboveGround_2_2t"] - g["heightAboveGround_2_2d"]
    gd["px_r850"] = g["isobaricInhPa_850_r"]
    gd["px_vrate"] = g["planetaryBoundaryLayer_0_VRATE"]
    gd["px_gustfac"] = g["surface_0_gust"] / np.maximum(ws10, eps)
    gd["px_dswrf"] = g["surface_0_dswrf"]
    gd["px_dlwrf"] = g["surface_0_dlwrf"]
    gd["px_tp"] = g["surface_0_tp"]
    for k, v in _veer(g["heightAboveGround_10_10u"], g["heightAboveGround_10_10v"],
                      g["heightAboveGround_100_100u"], g["heightAboveGround_100_100v"],
                      "px_veer_10_100").items():
        gd[k] = v
    for k, v in _veer(g["heightAboveGround_100_100u"], g["heightAboveGround_100_100v"],
                      g["isobaricInhPa_850_u"], g["isobaricInhPa_850_v"],
                      "px_veer_100_850").items():
        gd[k] = v
    gcols = [c for c in gd.columns if c.startswith("px_")]
    gagg = gd.groupby("forecast_kst_dtm")[gcols].agg(["mean", "std"])
    gagg.columns = [f"gfs_{c}_{s}" for c, s in gagg.columns]

    l = pd.read_csv(ldaps_path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    lws10 = _wind_speed(l["heightAboveGround_10_10u"], l["heightAboveGround_10_10v"])
    lws50 = _wind_speed(l["heightAboveGround_50_50MUmax"], l["heightAboveGround_50_50MVmax"])
    lws50min = _wind_speed(l["heightAboveGround_50_50MUmin"], l["heightAboveGround_50_50MVmin"])
    lblws = _wind_speed(l["heightAboveGround_5_XBLWS"], l["heightAboveGround_5_YBLWS"])
    ld = pd.DataFrame({"forecast_kst_dtm": l["forecast_kst_dtm"]})
    ld["px_l_alpha"] = np.log(np.maximum(lws50, eps) / np.maximum(lws10, eps)) / np.log(5.0)
    ld["px_l_ws117"] = lws50 * ((117.0 / 50.0) ** ld["px_l_alpha"])
    ld["px_l_ws50min"] = lws50min
    ld["px_l_ws50_range"] = lws50 - lws50min
    ld["px_l_gustfac"] = lws50 / np.maximum(lws10, eps)
    ld["px_l_blws_ratio"] = lblws / np.maximum(lws10, eps)
    ld["px_l_dpt_dep"] = l["heightAboveGround_2_t"] - l["heightAboveGround_2_dpt"]
    ld["px_l_q"] = l["heightAboveGround_2_q"]
    ld["px_l_prmsl"] = l["meanSea_0_prmsl"]
    ld["px_l_hcc"] = l["etc_0_hcc"]
    ld["px_l_mcc"] = l["etc_0_mcc"]
    ld["px_l_vlcdc"] = l["etc_0_VLCDC"]
    ld["px_l_ndnsw"] = l["surface_0_NDNSW"]
    ld["px_l_ndnlw"] = l["surface_0_NDNLW"]
    ld["px_l_lssrate"] = l["surface_0_lssrate"]
    ld["px_l_snol"] = l["surface_0_snol"]
    lcols = [c for c in ld.columns if c.startswith("px_")]
    lagg = ld.groupby("forecast_kst_dtm")[lcols].agg(["mean", "std"])
    lagg.columns = [f"ldaps_{c}_{s}" for c, s in lagg.columns]

    # 컬럼 집합은 데이터에 의존하지 않아야 한다 (train/test 정렬 보장)
    return gagg.join(lagg, how="outer").sort_index()


def build_features(ldaps_path, gfs_path, context: bool = True,
                   phys: bool = True) -> pd.DataFrame:
    """예보 대상 시각(forecast_kst_dtm) 인덱스의 피처 테이블 생성.

    phys=False 는 v5 이전(276피처) 레거시 캐시 재현용이다.
    """
    ldaps, davail = build_ldaps_features(ldaps_path)
    gfs = build_gfs_features(gfs_path)
    feat = ldaps.join(gfs, how="outer")
    feat = add_time_features(feat)
    if context:
        feat = add_context(feat, davail)
    if phys:
        feat = feat.join(build_phys_features(ldaps_path, gfs_path), how="left")
    return feat.sort_index()


if __name__ == "__main__":
    # 자체 점검: train/test 피처 컬럼이 어긋나면 추론이 조용히 깨진다.
    from config import TEST_DIR, TRAIN_DIR

    tr = build_features(TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv")
    te = build_features(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")
    assert list(tr.columns) == list(te.columns), "train/test 피처 컬럼 불일치"
    assert len(tr.columns) == 348, f"피처 수 {len(tr.columns)} != 348"
    for f in (tr, te):
        px = f[[c for c in f.columns if "_px_" in c]]
        assert np.isfinite(px.to_numpy(dtype=float)[~px.isna().to_numpy()]).all(), "물리 팩에 Inf"
        assert (px["gfs_px_ws117_mean"].dropna() > 0).all(), "허브고도 풍속이 음수"
    print(f"OK: train {tr.shape}, test {te.shape}, 컬럼 일치")
