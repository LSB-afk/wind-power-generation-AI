"""예측 후처리: 그룹별 (scale, floor).

평가가 '실제 발전량 >= 설비용량 10%' 시간대만 대상이므로,
낮은 예측값은 (a) 실제도 낮으면 평가 제외, (b) 실제가 높으면 큰 오차가 된다.
따라서 하한(floor)을 두면 평가 대상 시간대의 오차만 선택적으로 줄어든다.
scale은 분위 회귀의 잔여 저편향(고발전 구간 과소예측) 보정.
"""
import numpy as np

from metrics import group_ficr, group_nmae


def group_score(pred, actual, cap):
    return 0.5 * (1 - group_nmae(pred, actual, cap)) + 0.5 * group_ficr(pred, actual, cap)


def tune_post(pred, actual, cap):
    """홀드아웃 예측/실측으로 (scale, floor) 그리드서치 → (best_score, scale, floor)."""
    best = (-1.0, 1.0, 0.0)
    for sc in np.arange(0.90, 1.401, 0.025):
        for fl in np.arange(0.0, 0.201, 0.005) * cap:
            s = group_score(apply_post(pred, cap, sc, fl), actual, cap)
            if s > best[0]:
                best = (s, sc, fl)
    return best


def apply_post(pred, cap, scale, floor):
    return np.clip(np.maximum(pred * scale, floor), 0, cap)
