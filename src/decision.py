"""의사결정 이론 기반 점예측 — 예측분포에서 대회 점수 기대값 최대점을 고른다.

왜 필요한가. 점수식을 시간 단위로 분해하면 밴드 적중 1회의 가치가 NMAE 로
환산해 `a_t / ā` 배(정격 근처면 2.0 x 설비용량)에 달한다. 오차는 설비용량을
넘을 수 없으므로 **FICR 항이 사실상 의사결정을 지배**한다. 그런데 우리는
quantile(0.60), 즉 중앙값 근처를 예측해 왔다 — 중앙값은 NMAE 를 최소화하는 점이지
밴드 적중을 최대화하는 점이 아니다.

올바른 점은 폭 12%cap 창의 **발전량 가중 확률질량**이 최대인 지점이다.
다분위 예측으로 조건부 CDF 를 세우고, 후보 격자에서 기대 점수를 직접 계산해 고른다.

기대 점수(시간 단위, 상수배 무시):
    U(f) = -E[|f-a| ; a>=0.1C]/C  +  E[a * rate(|f-a|) ; a>=0.1C] / (4*ā)
    rate = 4 (|e|<=0.06C), 3 (0.06C<|e|<=0.08C), 0 (그 외)
평가에서 실발전 0.1C 미만은 제외되므로 적분 구간도 그 위로 제한한다.
ā 는 **학습 연도의** 평가 대상 평균 발전량이다(정답 미사용).
"""
import numpy as np

BAND1, BAND2 = 0.06, 0.08
EVAL_MIN = 0.10


def _segments(Q, taus):
    """분위 예측 → 구간별 (하한, 상한, 질량). 구간 내 밀도는 균일로 근사."""
    Q = np.sort(Q, axis=1)
    lo, hi = Q[:, :-1], Q[:, 1:]
    mass = np.diff(taus)[None, :]
    return lo, hi, np.broadcast_to(mass, lo.shape)


def _clip_stats(lo, hi, mass, a, b):
    """구간 [lo,hi] 와 [a,b] 교집합의 (질량, 질량x평균값). a,b 는 (n,1) 또는 스칼라."""
    olo = np.maximum(lo, a)
    ohi = np.minimum(hi, b)
    w = np.maximum(ohi - olo, 0.0)
    span = np.maximum(hi - lo, 1e-9)
    frac = w / span
    m = mass * frac
    return m, m * 0.5 * (olo + ohi)


def optimal_point(Q, taus, cap, abar, step=0.01):
    """시간별 기대 점수 최대점. Q:(n,K) 분위 예측(kWh).

    후보는 평가 가능 구간 [0.1C, C] 전역 격자를 쓴다. 중앙값 주변으로 제한하면
    최빈 구간이 멀리 있을 때 도달하지 못한다.
    """
    lo, hi, mass = _segments(Q, taus)
    n = Q.shape[0]
    emin = EVAL_MIN * cap
    cands = np.arange(emin, cap + 1e-9, step * cap)

    # 평가 대상(a >= 0.1C) 으로 적분 구간 제한
    lo_e = np.maximum(lo, emin)
    hi_e = np.maximum(hi, emin)

    best_u = np.full(n, -np.inf)
    best_f = np.full(n, emin)
    for c in cands:
        f = np.full((n, 1), c)
        # E|f-a| : 구간을 f 기준 좌/우로 나눠 사다리꼴 적분
        m_all, ma_all = _clip_stats(lo_e, hi_e, mass, emin, cap)
        m_l, ma_l = _clip_stats(lo_e, hi_e, mass, emin, f)
        m_r, ma_r = _clip_stats(lo_e, hi_e, mass, f, cap)
        eabs = ((f * m_l - ma_l) + (ma_r - f * m_r)).sum(axis=1)
        # 밴드별 발전량 가중 질량
        _, a6 = _clip_stats(lo_e, hi_e, mass, f - BAND1 * cap, f + BAND1 * cap)
        _, a8 = _clip_stats(lo_e, hi_e, mass, f - BAND2 * cap, f + BAND2 * cap)
        A6 = a6.sum(axis=1)
        A68 = a8.sum(axis=1) - A6
        u = -eabs / cap + (4.0 * A6 + 3.0 * A68) / (4.0 * abar)
        better = u > best_u
        best_u = np.where(better, u, best_u)
        best_f = np.where(better, f[:, 0], best_f)
    return best_f


def demo():
    """자체 점검: 좌측으로 치우친 분포에서 최적점이 최빈 쪽으로 이동하는지."""
    cap = 100.0
    taus = np.array([0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95])
    # 중앙값 아래는 성기고 위쪽(80~90)에 질량이 촘촘한 분포 — 창 모드가 중앙값과 멀다
    Q = np.tile(np.array([10., 20., 30., 35., 80., 85., 90.]), (200, 1))
    f = optimal_point(Q, taus, cap, abar=50.0)
    med = Q[0, 3]
    assert f[0] > med + 20, f"창 모드로 이동하지 않음: {f[0]} vs 중앙값 {med}"
    # 대칭 분포에서는 중앙값 근처를 유지해야 한다
    Qs = np.tile(np.array([30., 42., 48., 50., 52., 58., 70.]), (50, 1))
    fs = optimal_point(Qs, taus, cap, abar=50.0)
    assert abs(fs[0] - 50.0) <= 3.0, f"대칭 분포인데 중앙값에서 벗어남: {fs[0]}"
    print(f"decision demo 통과 ✓  좌편향 최적점 {f[0]:.1f} (중앙값 {med:.1f}), "
          f"대칭 최적점 {fs[0]:.1f}")


if __name__ == "__main__":
    demo()
