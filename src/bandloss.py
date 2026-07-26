"""밴드 포화 손실 — FICR 을 직접 겨냥한 LightGBM 커스텀 목적함수.

문제: Public 격차의 84% 가 FICR 이고, 잔차를 줄여 1등과 같은 정확도에 맞춰도
우리 FICR 은 0.414 로 1등의 0.465 에 못 미친다. 즉 **오차 크기가 아니라
오차 분포의 모양**이 다르다.

원인은 손실함수 불일치다. pinball 손실은 큰 오차도 끝까지 줄이려 하지만,
FICR 은 `|오차| <= 6%cap` 안에 들어갔는지만 본다 — 밴드 밖은 8%cap 를 넘는 순간
얼마나 크든 0원이다. 따라서 최적 전략은 **가망 없는 시간대를 포기하고
밴드 근처 시간대를 안으로 밀어 넣는 것**이고, 그러려면 큰 잔차에서
기울기가 사라지는 포화 손실이 필요하다.

구현: pinball(비대칭, 평가 필터 대응) + Welsch(포화, 밴드 대응) 혼합.
  Welsch:  L(r) = 1 - exp(-(r/c)^2),  c = 밴드 반폭(0.06 x 설비용량)
  기울기가 |r| >> c 에서 0 으로 수렴해 이상치를 자동으로 포기한다.

타깃·예측 모두 설비용량으로 정규화한 공간에서 계산한다.
"""
import numpy as np

BAND = 0.06                      # 정산 1구간 반폭 (설비용량 대비)
_WELSCH_PEAK = 0.857 / BAND      # |grad| 최댓값 — pinball 과 스케일을 맞추기 위함


def make_objective(w_band: float, alpha: float = 0.60, c: float = BAND):
    """LightGBM fobj. w_band=0 이면 순수 pinball, 1 이면 순수 밴드 포화."""
    peak = 0.857 / c

    def _grad_hess(y_true, y_pred):
        r = y_pred - y_true
        # pinball(alpha): 과소예측(r<0) 에 alpha, 과대예측에 (1-alpha) 벌점
        g_pin = np.where(r < 0.0, -alpha, 1.0 - alpha)
        # Welsch: 밴드 밖으로 멀어질수록 기울기가 0 으로 사라진다
        e = np.exp(-((r / c) ** 2))
        g_wel = (2.0 * r / c**2) * e / peak
        grad = (1.0 - w_band) * g_pin + w_band * g_wel
        # 헤시안은 상수로 둔다 — L1 계열 목적함수의 표준 처리
        return grad, np.ones_like(grad)

    def fobj(a, b):
        """LightGBM 버전에 따라 (y_true, y_pred) 또는 (y_pred, Dataset) 로 온다."""
        if hasattr(b, "get_label"):
            return _grad_hess(np.asarray(b.get_label(), dtype=float), np.asarray(a, dtype=float))
        return _grad_hess(np.asarray(a, dtype=float), np.asarray(b, dtype=float))

    return fobj


def band_eval(cap_ratio_true, cap_ratio_pred):
    """참고용: 정규화 공간에서의 밴드 적중률."""
    e = np.abs(cap_ratio_pred - cap_ratio_true)
    return float((e <= BAND).mean())


def demo():
    """자체 점검: 포화 손실이 밴드 적중률을 실제로 높이는지 (단순 1차원 문제)."""
    rng = np.random.default_rng(0)
    # 70% 는 좁게, 30% 는 아주 넓게 흩어진 이봉 분포 — 이상치를 쫓으면 손해
    y = np.concatenate([rng.normal(0.5, 0.03, 7000), rng.normal(0.5, 0.40, 3000)])
    grid = np.linspace(0.2, 0.8, 601)

    def total(f, w):
        r = f - y
        pin = np.maximum(0.60 * (-r), (0.60 - 1) * (-r))
        wel = 1 - np.exp(-((r / BAND) ** 2))
        return ((1 - w) * pin + w * wel).mean()

    f_pin = grid[np.argmin([total(f, 0.0) for f in grid])]
    f_band = grid[np.argmin([total(f, 1.0) for f in grid])]
    hit_pin, hit_band = band_eval(y, f_pin), band_eval(y, f_band)
    assert hit_band >= hit_pin, f"포화 손실이 밴드 적중을 못 높임 {hit_pin} vs {hit_band}"
    g, h = make_objective(0.5)(y, np.full_like(y, 0.5))
    assert np.all(np.isfinite(g)) and np.all(h > 0), "기울기/헤시안 이상"
    # 큰 잔차에서 기울기가 pinball 쪽만 남는지 (포화 확인)
    gf, _ = make_objective(1.0)(np.zeros(3), np.array([0.0, 0.06, 1.0]))
    assert abs(gf[2]) < 1e-6 < abs(gf[1]), f"포화가 작동하지 않음: {gf}"
    print(f"bandloss demo 통과 ✓  밴드적중 pinball {hit_pin:.3f} → 포화 {hit_band:.3f}")


if __name__ == "__main__":
    demo()
