"""전처리 재설계: 원본 기상변수 전체 x 전체 격자 피처화.

기존 파이프라인(v1~v8)은 격자별 피벗을 바람 변수에만 적용하고 나머지는
격자 평균으로 뭉갰다. LDAPS 16격자 x 28변수, GFS 9격자 x 33변수 중
실제로 쓰인 것은 20% 미만이다. 산악 능선의 지형 효과는 격자마다 다르므로
평균이 정보를 지운다는 가정 하에, 전 변수를 격자별로 펼친다.

Data Leakage 규정 준수: 원본 CSV의 예보값만 사용한다. 각 행은 이미
전일 13:00(KST) 발표분으로 한정 제공되므로 예측기준시점 규칙을 자동 충족하며,
격자/변수 간 결합만 할 뿐 시간 축을 넘나드는 연산은 하지 않는다.
"""
import numpy as np
import pandas as pd

from config import ROOT, TEST_DIR, TRAIN_DIR

CACHE = ROOT / "cache"
_META = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}


def _derive(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """격자 단위 물리 파생 — 풍속/시어/공기밀도 등."""
    out = pd.DataFrame(index=df.index)
    ws = lambda u, v: np.sqrt(df[u] ** 2 + df[v] ** 2)
    if source == "ldaps":
        out["ws10"] = ws("heightAboveGround_10_10u", "heightAboveGround_10_10v")
        out["ws50x"] = ws("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax")
        out["ws50n"] = ws("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin")
        out["ws50m"] = 0.5 * (out["ws50x"] + out["ws50n"])
        out["wsbl"] = ws("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS")
        # 전단지수는 10→50 m 로 추정하되, 외삽은 가장 가까운 층(50 m)에서 출발한다.
        # 10 m 를 외삽 기점으로 쓰면 고도비가 12배가 되어 alpha 오차가 증폭된다.
        out["alpha"] = np.log(np.maximum(out["ws50m"], 0.5) /
                              np.maximum(out["ws10"], 0.5)) / np.log(5.0)
        out["ws117"] = out["ws50m"] * (117.0 / 50.0) ** out["alpha"]
        out["turb"] = (out["ws50x"] - out["ws50n"]) / np.maximum(out["ws50m"], 0.5)
        t, p = df["heightAboveGround_2_t"], df["surface_0_sp"]
        out["rho"] = _rho(p, t, df["heightAboveGround_2_q"])
        # 공기밀도 보정 풍속 (IEC 61400-12-1): v_n = v * (rho/rho0)^(1/3)
        out["ws117n"] = out["ws117"] * (out["rho"] / 1.225) ** (1 / 3)
        out["pdens"] = 0.5 * out["rho"] * out["ws117"] ** 3
        # 허브고도가 경계층 안인지 밖인지 — 밖이면 지표 전단 관계가 깨진다
        out["z_blh"] = 117.0 / np.maximum(df["etc_0_blh"], 20.0)
        out["dir_sin"], out["dir_cos"] = _dirsc(df, "heightAboveGround_10_10u",
                                                "heightAboveGround_10_10v")
    else:
        out["ws10"] = ws("heightAboveGround_10_10u", "heightAboveGround_10_10v")
        out["ws80"] = ws("heightAboveGround_80_u", "heightAboveGround_80_v")
        out["ws100"] = ws("heightAboveGround_100_100u", "heightAboveGround_100_100v")
        out["wspbl"] = ws("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v")
        out["ws850"] = ws("isobaricInhPa_850_u", "isobaricInhPa_850_v")
        out["ws700"] = ws("isobaricInhPa_700_u", "isobaricInhPa_700_v")
        out["ws500"] = ws("isobaricInhPa_500_u", "isobaricInhPa_500_v")
        # 허브(117 m)를 감싸는 80/100 m 쌍으로 전단지수 추정 → 100 m 에서 17 m 만 외삽
        out["alpha"] = np.log(np.maximum(out["ws100"], 0.5) /
                              np.maximum(out["ws80"], 0.5)) / np.log(100.0 / 80.0)
        out["alpha_deep"] = np.log(np.maximum(out["ws100"], 0.5) /
                                   np.maximum(out["ws10"], 0.5)) / np.log(10.0)
        out["ws117"] = out["ws100"] * (1.17 ** out["alpha"])
        t, p = df["heightAboveGround_2_2t"], df["surface_0_sp"]
        out["rho"] = _rho(p, t, df["heightAboveGround_2_2sh"])
        out["ws117n"] = out["ws117"] * (out["rho"] / 1.225) ** (1 / 3)
        out["pdens"] = 0.5 * out["rho"] * out["ws117"] ** 3
        out["gustfac"] = df["surface_0_gust"] / np.maximum(out["ws10"], 0.5)
        # 정적 안정도 (하층 기온감률) — 전단지수를 좌우
        out["lapse"] = df["heightAboveGround_2_2t"] - df["isobaricInhPa_850_t"]
        # 벌크 리처드슨 수: 부력/전단 비. 양수 크면 안정(전단 강), 음수면 불안정(혼합)
        dth = df["isobaricInhPa_850_t"] - df["heightAboveGround_2_2t"]
        dv2 = np.maximum((out["ws100"] - out["ws10"]) ** 2, 0.25)
        out["rib"] = (9.81 / t) * dth * 1400.0 / dv2
        out["dir_sin"], out["dir_cos"] = _dirsc(df, "heightAboveGround_100_100u",
                                                "heightAboveGround_100_100v")
    return out


def _rho(p, t, q):
    """습윤공기 밀도 — 가온도(virtual temperature) 사용. q는 비습(kg/kg)."""
    return p / (287.05 * t * (1.0 + 0.608 * np.clip(q, 0, 0.05)))


def _dirsc(df, u, v):
    th = np.arctan2(df[v], df[u])
    return np.sin(th), np.cos(th)


def build_full(path, source: str) -> pd.DataFrame:
    """예보 대상 시각 x (전 변수 x 전 격자 + 격자 집계) 피처 테이블."""
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    raw_cols = [c for c in df.columns if c not in _META]
    df = pd.concat([df[["forecast_kst_dtm", "grid_id"] + raw_cols],
                    _derive(df, source)], axis=1)
    val_cols = [c for c in df.columns if c not in ("forecast_kst_dtm", "grid_id")]

    wide = df.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=val_cols)
    wide.columns = [f"{source[0]}{int(g):02d}_{v}" for v, g in wide.columns]

    agg = df.groupby("forecast_kst_dtm")[val_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{source}_{c}_{s}" for c, s in agg.columns]
    # 격자 간 편차 = 예보 불확실성 대리 지표
    return wide.join(agg)


def build(ldaps_path, gfs_path) -> pd.DataFrame:
    feat = build_full(ldaps_path, "ldaps").join(build_full(gfs_path, "gfs"), how="outer")
    idx = feat.index
    feat["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feat["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    feat["month"] = idx.month
    # 발표분 기준 예보 선행시간 (01시=12h ... 익일 00시=35h)
    feat["lead_h"] = np.where(idx.hour == 0, 35, idx.hour + 11)
    # 두 예보원 불일치 = 예보 신뢰도 대리 지표
    feat["x_ws117_diff"] = feat["ldaps_ws117_mean"] - feat["gfs_ws117_mean"]
    feat["x_ws10_diff"] = feat["ldaps_ws10_mean"] - feat["gfs_ws10_mean"]
    return feat.sort_index()


def main():
    CACHE.mkdir(exist_ok=True)
    for tag, (lp, gp) in {
        "train": (TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv"),
        "test": (TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv"),
    }.items():
        out = CACHE / f"full_{tag}.parquet"
        if out.exists():
            print(f"{tag}: 캐시 존재, 건너뜀")
            continue
        f = build(lp, gp)
        f.to_parquet(out)
        print(f"{tag}: {f.shape[0]:,}행 x {f.shape[1]:,}피처 → {out.name}")


if __name__ == "__main__":
    main()
