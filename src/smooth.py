"""예보 블록 내 시간 평활 — 시간별 독립 예측의 잡음 제거.

각 시간을 독립적으로 예측하면 GBDT 특유의 계단형 잡음이 붙는다. 그러나 실제
발전량은 기상 예보 궤적을 따라 매끄럽게 움직인다. 인접 시간 예측을 평균하면
이 잡음이 상쇄되어 오차가 줄고, 그만큼 6%밴드 적중이 올라간다.

**누수 차단**: 평활은 반드시 같은 발표분(data_available) 블록 안에서만 한다.
블록은 01:00~익일 00:00 이며, 익일 01:00 값은 하루 뒤 13:00 에 발표된 예보에서
나온다. 00:00 예측에 그 값을 섞으면 예측기준시점 이후 정보를 쓰는 것이 된다.
"""
import numpy as np
import pandas as pd


def block_id(index: pd.DatetimeIndex) -> np.ndarray:
    """발표분 블록 ID. 01:00~익일 00:00 이 한 블록이므로 1시간 당겨 날짜를 취한다."""
    return (index - pd.Timedelta(hours=1)).normalize().values


def smooth_within_block(pred: np.ndarray, index: pd.DatetimeIndex, k: int) -> np.ndarray:
    """블록 내 중앙 이동평균. 블록 경계에서는 사용 가능한 시간만 평균한다."""
    if k <= 1:
        return pred.astype(float)
    s = pd.Series(np.asarray(pred, dtype=float), index=index)
    g = s.groupby(block_id(index))
    return g.transform(
        lambda x: x.rolling(k, center=True, min_periods=1).mean()).to_numpy()


def demo():
    """자체 점검: 블록 경계를 넘지 않는지, 잡음이 실제로 줄어드는지."""
    idx = pd.date_range("2025-01-01 01:00", periods=48, freq="h")
    b = block_id(idx)
    assert len(np.unique(b)) == 2, f"블록이 2개여야 함: {np.unique(b)}"
    assert (b[:24] == b[0]).all() and (b[24:] == b[24]).all(), "블록 경계가 어긋남"

    # 블록마다 값이 확 다르면, 평활해도 서로 섞이지 않아야 한다
    v = np.r_[np.zeros(24), np.ones(24) * 100.0]
    sm = smooth_within_block(v, idx, 3)
    assert sm[:24].max() == 0.0 and sm[24:].min() == 100.0, f"블록 간 누수: {sm[22:26]}"

    # 매끄러운 신호 + 잡음 → 평활이 오차를 줄여야 한다
    rng = np.random.default_rng(0)
    idx2 = pd.date_range("2025-01-01 01:00", periods=24 * 30, freq="h")
    truth = 50 + 30 * np.sin(np.arange(len(idx2)) / 8.0)
    noisy = truth + rng.normal(0, 8, len(idx2))
    sm2 = smooth_within_block(noisy, idx2, 3)
    e0, e1 = np.abs(noisy - truth).mean(), np.abs(sm2 - truth).mean()
    assert e1 < e0, f"평활이 오차를 못 줄임 {e0:.2f} → {e1:.2f}"
    print(f"smooth demo 통과 ✓  블록 격리 확인, 잡음 MAE {e0:.2f} → {e1:.2f}")


if __name__ == "__main__":
    demo()
