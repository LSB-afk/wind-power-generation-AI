"""대회 평가지표: NMAE, FICR(정산금획득률), 총점.

총점 = 0.5 x (1 - NMAE) + 0.5 x FICR
평가는 실제 발전량이 설비용량의 10% 이상인 시간대만 대상.
"""
import numpy as np

from config import CAPACITY_KWH, EVAL_THRESHOLD_RATIO, GROUPS, MAX_SETTLEMENT_RATE, SETTLEMENT_TIERS


def _eval_mask(actual: np.ndarray, capacity: float) -> np.ndarray:
    return actual >= EVAL_THRESHOLD_RATIO * capacity


def group_nmae(pred: np.ndarray, actual: np.ndarray, capacity: float) -> float:
    """평가 대상 시간대의 mean(|pred - actual| / 설비용량)."""
    mask = _eval_mask(actual, capacity)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred[mask] - actual[mask]) / capacity))


def group_ficr(pred: np.ndarray, actual: np.ndarray, capacity: float) -> float:
    """획득 정산금 / 이론상 최대 정산금.

    시간별 예측오차율 e = |pred - actual| / 설비용량 * 100 (%)
    e <= 6%  → 4원/kWh x 실제 발전량
    e <= 8%  → 3원/kWh x 실제 발전량
    e >  8%  → 0원
    이론상 최대 = 4원/kWh x 실제 발전량
    """
    mask = _eval_mask(actual, capacity)
    if mask.sum() == 0:
        return np.nan
    p, a = pred[mask], actual[mask]
    err_pct = np.abs(p - a) / capacity * 100.0
    rate = np.zeros_like(err_pct)
    for threshold, won in reversed(SETTLEMENT_TIERS):
        rate = np.where(err_pct <= threshold, won, rate)
    earned = float(np.sum(rate * a))
    theoretical_max = float(np.sum(MAX_SETTLEMENT_RATE * a))
    return earned / theoretical_max


def competition_score(preds: dict, actuals: dict) -> dict:
    """그룹별 pred/actual dict를 받아 총점과 세부 지표를 반환."""
    nmaes, ficrs, detail = [], [], {}
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        nm = group_nmae(preds[g], actuals[g], cap)
        fi = group_ficr(preds[g], actuals[g], cap)
        detail[g] = {"nmae": nm, "ficr": fi}
        nmaes.append(nm)
        ficrs.append(fi)
    mean_nmae = float(np.nanmean(nmaes))
    mean_ficr = float(np.nanmean(ficrs))
    score = 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr
    return {"score": score, "one_minus_nmae": 1.0 - mean_nmae, "ficr": mean_ficr, "detail": detail}
