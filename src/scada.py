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


def build_bad_mask(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨 인덱스(kst_dtm) 기준 그룹별 '{group}_bad' 불리언 마스크."""
    scada = {
        "vestas": pd.read_csv(TRAIN_DIR / "scada_vestas_train.csv", encoding="utf-8-sig",
                              parse_dates=["kst_dtm"]).set_index("kst_dtm").sort_index(),
        "unison": pd.read_csv(TRAIN_DIR / "scada_unison_train.csv", encoding="utf-8-sig",
                              parse_dates=["kst_dtm"]).set_index("kst_dtm").sort_index(),
    }
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        agg = _hourly(scada[maker], maker, turbines)
        j = labels[[g]].join(agg, how="left")
        mismatch = (np.abs(j[g] - j["scada_kwh"]) / CAPACITY_KWH[g] > 0.05).fillna(False)
        out[f"{g}_bad"] = (j["n_stopped"] >= 1.0).fillna(False) | mismatch
    return pd.DataFrame(out, index=labels.index)
