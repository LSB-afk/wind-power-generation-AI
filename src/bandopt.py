"""FICR 직접 최적화 — 예측 수준별 오프셋 테이블.

문제 정의: Public 격차의 84%가 FICR 이고 NMAE 는 이미 1등의 92% 수준이다.
1등은 더 정확한 게 아니라 같은 정확도에서 정산금을 12.5% 더 뽑는다.

이유는 손실함수 불일치다. 우리는 quantile(0.60) 로 중앙값 계열을 예측하는데,
대회 점수의 FICR 항은 `|오차| <= 6%cap` 이라는 **밴드 적중**에 지급된다.
밴드 적중 확률을 최대화하는 점은 중앙값이 아니라 폭 12%cap 창의 확률질량이
최대인 지점(창 모드)이다. 게다가 실발전 10%cap 미만은 채점에서 빠지므로
낮은 예측 구간에서는 `E[a | a >= 0.10cap]` 이 옳은 목표가 된다.

접근: 예측 수준별로 오프셋 delta(bin) 을 두고, 실제 대회 점수를 최대화하도록
격자 탐색한다. 오프셋은 **예측값만 보고 정해지므로** 추론 시 정답을 쓰지 않는다.
검증은 한 폴드에서 적합해 다른 폴드에 적용하는 교차 전이로 한다 — 연도가 바뀌어도
살아남는지가 유일한 채택 근거다.
"""
import numpy as np

from config import CAPACITY_KWH

EDGES = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.01])
DELTAS = np.arange(-0.06, 0.0601, 0.005)
MIN_ROWS = 150


def score_parts(pred, act, cap):
    """평가 대상 시간대의 (NMAE 항, FICR 분자, FICR 분모)."""
    m = act >= 0.10 * cap
    p, a = pred[m], act[m]
    e = np.abs(p - a) / cap
    r = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
    return e, r * a, 4.0 * a


def group_score(pred, act, cap):
    e, earn, mx = score_parts(pred, act, cap)
    if len(e) == 0:
        return np.nan
    return 0.5 * (1 - e.mean()) + 0.5 * earn.sum() / mx.sum()


def level_bin(pred, cap):
    """예측 수준(정격 대비)별 구간 인덱스."""
    return np.clip(np.digitize(pred / cap, EDGES) - 1, 0, len(EDGES) - 2)


def fit_offsets(pred, act, cap):
    """구간별 오프셋을 좌표하강으로 탐색해 그룹 점수를 최대화한다.

    한 구간의 오프셋은 그 구간 행의 오차만 바꾸지만, FICR 은 전체 합에 대한
    비율이라 구간끼리 약하게 결합된다. 그래서 전역 점수를 목적으로 순회한다.
    """
    b = level_bin(pred, cap)
    off = np.zeros(len(EDGES) - 1)
    best = group_score(apply_offsets(pred, cap, off), act, cap)
    for _ in range(3):                       # 좌표하강 3회면 수렴한다
        for i in range(len(off)):
            if (b == i).sum() < MIN_ROWS:
                continue
            cur = off[i]
            for d in DELTAS:
                off[i] = d
                s = group_score(apply_offsets(pred, cap, off), act, cap)
                if s > best:
                    best, cur = s, d
            off[i] = cur
    return off, best


def apply_offsets(pred, cap, off):
    b = level_bin(pred, cap)
    return np.clip(pred + off[b] * cap, 0.10 * cap, cap)


def demo():
    """자체 점검: 오프셋 적용이 밴드 적중을 실제로 늘리는지."""
    rng = np.random.default_rng(0)
    cap = 21600.0
    act = np.clip(rng.beta(2, 2, 20000) * cap, 0, cap)
    # 일부러 3%cap 하향 편향된 예측을 만든다 → 오프셋이 +0.03 근처를 찾아야 한다
    pred = np.clip(act - 0.03 * cap + rng.normal(0, 0.05 * cap, len(act)), 0, cap)
    off, s = fit_offsets(pred, act, cap)
    base = group_score(np.clip(pred, 0.10 * cap, cap), act, cap)
    assert s > base, f"오프셋이 점수를 못 올림 {base:.4f} → {s:.4f}"
    assert off[2:6].mean() > 0.01, f"하향 편향인데 양의 오프셋을 못 찾음: {off}"
    print(f"bandopt demo 통과 ✓  {base:.4f} → {s:.4f}, 오프셋 {np.round(off,3)}")


if __name__ == "__main__":
    demo()
