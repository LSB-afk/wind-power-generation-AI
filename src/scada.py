"""SCADA 기반 라벨 클리닝과 가용 발전량 복원.

SCADA 10분 데이터를 시간별 그룹 에너지로 집계해 라벨과 대조한다.
(power_kw10m 컬럼은 실측 확인 결과 10분 에너지(kWh)로, 시간 합=라벨과 corr 0.999)

'나쁜 시간대' = 학습에서 제외할 라벨 노이즈:
- 터빈 정지: 자체 풍속 >= 5 m/s 인데 출력이 없는 터빈이 시간 평균 1대 이상
  (정비/트립 시간대 — 기상만으로 설명 불가능한 저출력이라 기상→발전 학습을 오염)
- 계측 불일치: |라벨 - SCADA 합| > 설비용량의 5%

마스크는 학습 행 필터링에만 사용하며 검증/추론에는 사용하지 않는다 (누수 없음).
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from config import CAPACITY_KWH, TRAIN_DIR

_GROUP_TURBINES = {
    "kpx_group_1": ("vestas", range(1, 7)),
    "kpx_group_2": ("vestas", range(7, 13)),
    "kpx_group_3": ("unison", range(1, 6)),
}
_MIN_HEALTHY_TURBINES = {
    "kpx_group_1": 3,
    "kpx_group_2": 3,
    "kpx_group_3": 2,
}


@dataclass(frozen=True)
class WeightCalibration:
    """Immutable evidence for one fold-safe turbine-weight calibration."""

    group: str
    train_label_years: tuple[int, ...]
    turbine_columns: tuple[str, ...]
    weights: tuple[float, ...]
    calibration_row_count: int
    calibration_index_hash: str
    weights_hash: str


def _hash_index(index: pd.Index) -> str:
    digest = hashlib.sha256()
    digest.update(str(index.dtype).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(index, index=False).values.tobytes())
    return digest.hexdigest()


def _hash_weights(columns: tuple[str, ...], weights: tuple[float, ...]) -> str:
    payload = json.dumps(
        {"columns": list(columns), "weights": list(weights)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _turbine_columns(
    maker: str, turbines: Iterable[int]
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    ordered_turbines = tuple(int(turbine) for turbine in turbines)
    if not ordered_turbines:
        raise ValueError("at least one turbine is required")
    power_columns = tuple(
        f"{maker}_wtg{turbine:02d}_power_kw10m" for turbine in ordered_turbines
    )
    wind_columns = tuple(f"{maker}_wtg{turbine:02d}_ws" for turbine in ordered_turbines)
    return ordered_turbines, power_columns, wind_columns


def _sanitized_turbine_frames(
    scada: pd.DataFrame, maker: str, turbines: Iterable[int]
) -> tuple[tuple[int, ...], tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    ordered_turbines, power_columns, wind_columns = _turbine_columns(maker, turbines)
    power = scada.loc[:, list(power_columns)].copy()
    wind = scada.loc[:, list(wind_columns)].copy()
    power = power.where((power >= -100.0) & (power <= 800.0))
    wind = wind.where((wind >= 0.0) & (wind <= 60.0))
    return ordered_turbines, power_columns, power, wind


def estimate_turbine_weights(
    scada: pd.DataFrame,
    *,
    group: str,
    maker: str,
    turbines: Iterable[int],
    capacity_kwh: float,
    train_label_years: tuple[int, ...],
) -> WeightCalibration:
    """Estimate robust relative turbine shares from training label-years only."""
    years = tuple(sorted({int(year) for year in train_label_years}))
    if not years:
        raise ValueError("at least one training label-year is required")
    _, power_columns, power, wind = _sanitized_turbine_frames(scada, maker, turbines)
    hour_end = scada.index.ceil("h")
    group_power = power.sum(axis=1, min_count=len(power_columns))
    calibration_mask = (
        hour_end.year.isin(years)
        & power.notna().all(axis=1).to_numpy()
        & wind.notna().all(axis=1).to_numpy()
        & (power > 1.0).all(axis=1).to_numpy()
        & (wind >= 5.0).all(axis=1).to_numpy()
        & (group_power >= 0.10 * float(capacity_kwh) / 6.0).to_numpy()
    )
    calibration_power = power.loc[calibration_mask].sort_index()
    if calibration_power.empty:
        raise ValueError("no rows satisfy turbine-weight calibration requirements")

    row_total = calibration_power.sum(axis=1)
    weights = len(power_columns) * calibration_power.div(row_total, axis=0).median(
        axis=0
    )
    weights = weights / weights.mean()
    weight_tuple = tuple(float(value) for value in weights.to_numpy())
    if not np.isfinite(weight_tuple).all() or not (np.asarray(weight_tuple) > 0).all():
        raise ValueError("turbine-weight calibration produced invalid weights")
    index_hash = _hash_index(calibration_power.index)
    return WeightCalibration(
        group=str(group),
        train_label_years=years,
        turbine_columns=power_columns,
        weights=weight_tuple,
        calibration_row_count=len(calibration_power),
        calibration_index_hash=index_hash,
        weights_hash=_hash_weights(power_columns, weight_tuple),
    )


def reconstruct_weighted_potential(
    scada: pd.DataFrame,
    calibration: WeightCalibration,
    *,
    group: str,
    maker: str,
    turbines: Iterable[int],
    capacity_kwh: float,
    min_healthy: int,
) -> pd.Series:
    """Reconstruct all-turbine 10-minute output using weighted coverage."""
    _, power_columns, power, wind = _sanitized_turbine_frames(scada, maker, turbines)
    if calibration.group != group:
        raise ValueError("calibration group does not match reconstruction")
    if power_columns != calibration.turbine_columns:
        raise ValueError("calibration turbine columns do not match reconstruction")
    weights = np.asarray(calibration.weights, dtype=float)
    if len(weights) != len(power_columns) or not np.isfinite(weights).all():
        raise ValueError("calibration weights are invalid")
    if calibration.weights_hash != _hash_weights(
        power_columns, tuple(float(value) for value in weights)
    ):
        raise ValueError("calibration weights hash does not match weights")
    if min_healthy < 1 or min_healthy > len(power_columns):
        raise ValueError("minimum healthy turbine count is invalid")

    healthy = (
        (power.to_numpy() > 1.0) | (wind.to_numpy() < 5.0)
    ) & power.notna().to_numpy()
    healthy_count = healthy.sum(axis=1)
    healthy_output = np.where(
        healthy, np.nan_to_num(power.to_numpy(), nan=0.0), 0.0
    ).sum(axis=1)
    covered_weight = np.where(healthy, weights, 0.0).sum(axis=1)
    potential = np.full(len(scada), np.nan, dtype=float)
    valid = (healthy_count >= min_healthy) & (covered_weight > 0.0)
    potential[valid] = healthy_output[valid] * weights.sum() / covered_weight[valid]
    potential = np.clip(potential, 0.0, float(capacity_kwh) / 6.0)
    return pd.Series(potential, index=scada.index, name="weighted_potential_10m")


def build_weighted_targets(
    label_index: pd.Index,
    *,
    train_label_years: tuple[int, ...],
    target_label_years: tuple[int, ...] | None = None,
    groups: tuple[str, ...] | None = None,
    scada_frames: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, WeightCalibration]]:
    """Build fold-scoped hourly targets without accepting actual label values."""
    labels = pd.DatetimeIndex(label_index)
    selected_groups = tuple(_GROUP_TURBINES) if groups is None else tuple(groups)
    unknown = set(selected_groups) - set(_GROUP_TURBINES)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    training_years = tuple(sorted({int(year) for year in train_label_years}))
    years = (
        training_years
        if target_label_years is None
        else tuple(sorted({int(year) for year in target_label_years}))
    )
    if not set(years).issubset(training_years):
        raise ValueError("target label-years must be a subset of training label-years")
    frames = _load_scada() if scada_frames is None else dict(scada_frames)
    targets: dict[str, pd.Series] = {}
    calibrations: dict[str, WeightCalibration] = {}
    for group in selected_groups:
        maker, turbine_range = _GROUP_TURBINES[group]
        turbines = tuple(turbine_range)
        scada = frames[maker]
        calibration = estimate_turbine_weights(
            scada,
            group=group,
            maker=maker,
            turbines=turbines,
            capacity_kwh=CAPACITY_KWH[group],
            train_label_years=training_years,
        )
        potential_10m = reconstruct_weighted_potential(
            scada,
            calibration,
            group=group,
            maker=maker,
            turbines=turbines,
            capacity_kwh=CAPACITY_KWH[group],
            min_healthy=_MIN_HEALTHY_TURBINES[group],
        )
        hour_end = potential_10m.index.ceil("h")
        hourly = potential_10m.groupby(hour_end).mean() * 6.0
        hourly = hourly.reindex(labels)
        hourly = hourly.where(labels.year.isin(years))
        targets[f"{group}_weighted_potential"] = hourly
        calibrations[group] = calibration
    return pd.DataFrame(targets, index=labels), calibrations


def _hourly(d: pd.DataFrame, prefix: str, turbines) -> pd.DataFrame:
    pw = d[[f"{prefix}_wtg{t:02d}_power_kw10m" for t in turbines]]
    ws = d[[f"{prefix}_wtg{t:02d}_ws" for t in turbines]]
    # 물리 범위 밖 센티널(예: ±5천만) 제거 — 10분 에너지 상한 800 kWh
    pw = pw.where((pw >= -100) & (pw <= 800))
    ws = ws.where((ws >= 0) & (ws <= 60))
    hour_end = d.index.ceil("h")
    n = len(list(turbines))
    energy = pw.sum(axis=1, min_count=n // 2).groupby(hour_end).sum(min_count=3)
    stopped = pd.Series(
        ((ws.values >= 5.0) & ~(pw.values > 1)).sum(axis=1), index=d.index
    )
    n_stopped = stopped.groupby(hour_end).mean()
    return pd.DataFrame({"scada_kwh": energy, "n_stopped": n_stopped})


def _load_scada():
    return {
        "vestas": pd.read_csv(
            TRAIN_DIR / "scada_vestas_train.csv",
            encoding="utf-8-sig",
            parse_dates=["kst_dtm"],
        )
        .set_index("kst_dtm")
        .sort_index(),
        "unison": pd.read_csv(
            TRAIN_DIR / "scada_unison_train.csv",
            encoding="utf-8-sig",
            parse_dates=["kst_dtm"],
        )
        .set_index("kst_dtm")
        .sort_index(),
    }


def build_potential(labels: pd.DataFrame) -> pd.DataFrame:
    """가용률 보정 발전량(potential) 재구성.

    정지 터빈 시간대를 버리는 대신, 10분 단위로
    potential = (정상 터빈 에너지 합 / 정상 대수) x 전체 대수
    로 환산해 '전 터빈 가동 시 발전량'을 근사한다 (그룹 내 동일 기종 전제).

    정상 터빈 = 출력 > 1 kWh 또는 자체 풍속 < 5 m/s (저풍속 무출력은 정상).
    정상 대수 < 전체의 절반이면 NaN (신뢰 불가). 라벨-SCADA 불일치 시간대는
    별도 마스크(build_bad_mask)로 계속 제외한다.
    """
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        d = scada[maker]
        n = len(list(turbines))
        pw = d[[f"{maker}_wtg{t:02d}_power_kw10m" for t in turbines]]
        ws = d[[f"{maker}_wtg{t:02d}_ws" for t in turbines]]
        pw = pw.where((pw >= -100) & (pw <= 800))
        ws = ws.where((ws >= 0) & (ws <= 60))
        ok = (pw.values > 1) | (ws.values < 5.0)
        ok &= pw.notna().values
        e_ok = np.where(ok, np.nan_to_num(pw.values, nan=0.0), 0.0).sum(axis=1)
        n_ok = ok.sum(axis=1)
        pot10 = np.where(n_ok >= max(n // 2, 2), e_ok / np.maximum(n_ok, 1) * n, np.nan)
        cap10 = CAPACITY_KWH[g] / 6.0
        pot10 = np.clip(pot10, 0, cap10)
        hour_end = d.index.ceil("h")
        pot = pd.Series(pot10, index=d.index).groupby(hour_end).mean() * 6.0
        out[f"{g}_potential"] = pot.reindex(labels.index)
    return pd.DataFrame(out, index=labels.index)


def build_mismatch_mask(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨-SCADA 계측 불일치(>5%cap) 마스크 — potential 라벨 사용 시 제외 대상."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        agg = _hourly(scada[maker], maker, turbines)
        j = labels[[g]].join(agg, how="left")
        out[f"{g}_mismatch"] = (
            np.abs(j[g] - j["scada_kwh"]) / CAPACITY_KWH[g] > 0.05
        ).fillna(False)
    return pd.DataFrame(out, index=labels.index)


def build_bad_mask(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨 인덱스(kst_dtm) 기준 그룹별 '{group}_bad' 불리언 마스크."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        agg = _hourly(scada[maker], maker, turbines)
        j = labels[[g]].join(agg, how="left")
        mismatch = (np.abs(j[g] - j["scada_kwh"]) / CAPACITY_KWH[g] > 0.05).fillna(
            False
        )
        out[f"{g}_bad"] = (j["n_stopped"] >= 1.0).fillna(False) | mismatch
    return pd.DataFrame(out, index=labels.index)
