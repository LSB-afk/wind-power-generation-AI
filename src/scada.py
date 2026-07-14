"""SCADA 기반 라벨 클리닝 마스크 생성.

SCADA 10분 데이터를 시간별 그룹 에너지로 집계해 라벨과 대조한다.
(power_kw10m 컬럼은 실측 확인 결과 10분 에너지(kWh)로, 시간 합=라벨과 corr 0.999)

'나쁜 시간대' = 학습에서 제외할 라벨 노이즈:
- 터빈 정지: 자체 풍속 >= 5 m/s 인데 출력이 없는 터빈이 시간 평균 1대 이상
  (정비/트립 시간대 — 기상만으로 설명 불가능한 저출력이라 기상→발전 학습을 오염)
- 계측 불일치: |라벨 - SCADA 합| > 설비용량의 5%

마스크는 학습 행 필터링에만 사용하며 검증/추론에는 사용하지 않는다 (누수 없음).
"""
import numpy as np
import pandas as pd

from config import CAPACITY_KWH, TRAIN_DIR

_GROUP_TURBINES = {
    "kpx_group_1": ("vestas", range(1, 7)),
    "kpx_group_2": ("vestas", range(7, 13)),
    "kpx_group_3": ("unison", range(1, 6)),
}


def _hourly(d: pd.DataFrame, prefix: str, turbines) -> pd.DataFrame:
    pw = d[[f"{prefix}_wtg{t:02d}_power_kw10m" for t in turbines]]
    ws = d[[f"{prefix}_wtg{t:02d}_ws" for t in turbines]]
    # 물리 범위 밖 센티널(예: ±5천만) 제거 — 10분 에너지 상한 800 kWh
    pw = pw.where((pw >= -100) & (pw <= 800))
    ws = ws.where((ws >= 0) & (ws <= 60))
    hour_end = d.index.ceil("h")
    n = len(list(turbines))
    energy = pw.sum(axis=1, min_count=n // 2).groupby(hour_end).sum(min_count=3)
    stopped = pd.Series(((ws.values >= 5.0) & ~(pw.values > 1)).sum(axis=1), index=d.index)
    n_stopped = stopped.groupby(hour_end).mean()
    return pd.DataFrame({"scada_kwh": energy, "n_stopped": n_stopped})


def _load_scada():
    return {
        "vestas": pd.read_csv(TRAIN_DIR / "scada_vestas_train.csv", encoding="utf-8-sig",
                              parse_dates=["kst_dtm"]).set_index("kst_dtm").sort_index(),
        "unison": pd.read_csv(TRAIN_DIR / "scada_unison_train.csv", encoding="utf-8-sig",
                              parse_dates=["kst_dtm"]).set_index("kst_dtm").sort_index(),
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


def build_group_ws(index: pd.Index) -> pd.DataFrame:
    """그룹별 시간 평균 실측 풍속(SCADA) — 풍속 보정 모델의 학습 타깃."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        d = scada[maker]
        ws = d[[f"{maker}_wtg{t:02d}_ws" for t in turbines]]
        ws = ws.where((ws >= 0) & (ws <= 60))
        hour_end = d.index.ceil("h")
        out[f"{g}_ws"] = ws.mean(axis=1).groupby(hour_end).mean().reindex(index)
    return pd.DataFrame(out, index=index)


def build_mismatch_mask(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨-SCADA 계측 불일치(>5%cap) 마스크 — potential 라벨 사용 시 제외 대상."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        agg = _hourly(scada[maker], maker, turbines)
        j = labels[[g]].join(agg, how="left")
        out[f"{g}_mismatch"] = (np.abs(j[g] - j["scada_kwh"]) / CAPACITY_KWH[g] > 0.05).fillna(False)
    return pd.DataFrame(out, index=labels.index)


def build_bad_mask(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨 인덱스(kst_dtm) 기준 그룹별 '{group}_bad' 불리언 마스크."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        agg = _hourly(scada[maker], maker, turbines)
        j = labels[[g]].join(agg, how="left")
        mismatch = (np.abs(j[g] - j["scada_kwh"]) / CAPACITY_KWH[g] > 0.05).fillna(False)
        out[f"{g}_bad"] = (j["n_stopped"] >= 1.0).fillna(False) | mismatch
    return pd.DataFrame(out, index=labels.index)
