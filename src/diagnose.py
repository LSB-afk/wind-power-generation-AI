"""진단: 검증 점수의 노이즈 하한과 오차 구조.

시간별 풍속은 종관규모(2~5일)로 자기상관되어 있어 iid 부트스트랩은 SE를 과소평가한다.
7일(168h) 블록 부트스트랩으로 홀드아웃 점수의 표준오차를 추정하고,
그동안 채택했던 개선폭이 그 노이즈를 넘었는지 페어드 검정한다.
"""
import numpy as np

from config import CAPACITY_KWH, GROUPS

BLOCK = 168          # 7일 — 종관 스케일보다 길게
N_BOOT = 2000
RNG = np.random.default_rng(0)


def score_parts(pred, actual, cap):
    """평가 대상(실발전>=10%cap) 시간대의 시간별 기여값 반환."""
    m = actual >= 0.10 * cap
    p, a = pred[m], actual[m]
    err = np.abs(p - a) / cap
    rate = np.where(err <= 0.06, 4.0, np.where(err <= 0.08, 3.0, 0.0))
    return err, rate * a, 4.0 * a, m


def group_score_from(err, earn, mx):
    return 0.5 * (1 - err.mean()) + 0.5 * (earn.sum() / mx.sum())


def block_idx(n, rng):
    """길이 n을 덮는 블록 부트스트랩 인덱스."""
    starts = rng.integers(0, max(n - BLOCK, 1), size=n // BLOCK + 1)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def total_score(packs, key):
    """packs: {group: (pred, actual)} → 대회 총점."""
    nm, fi = [], []
    for g in GROUPS:
        pred, act = packs[g]
        err, earn, mx, _ = score_parts(pred, act, CAPACITY_KWH[g])
        nm.append(err.mean())
        fi.append(earn.sum() / mx.sum())
    return 0.5 * (1 - np.mean(nm)) + 0.5 * np.mean(fi)


def bootstrap_scores(packs_a, packs_b=None):
    """블록 부트스트랩. packs_b가 있으면 (a, b, a-b) 분포를 같은 리샘플로 페어드 계산."""
    n = len(packs_a[GROUPS[0]][0])
    out_a, out_b = [], []
    for _ in range(N_BOOT):
        idx = block_idx(n, RNG)
        out_a.append(total_score({g: (packs_a[g][0][idx], packs_a[g][1][idx]) for g in GROUPS}, None))
        if packs_b is not None:
            out_b.append(total_score({g: (packs_b[g][0][idx], packs_b[g][1][idx]) for g in GROUPS}, None))
    return np.array(out_a), (np.array(out_b) if packs_b is not None else None)


def load_pack(name, fold="fold24"):
    d = np.load(f"cache/{name}.npz", allow_pickle=True)
    return {g: (d[f"val|{fold}|{g}"], d[f"act|{fold}|{g}"]) for g in GROUPS}


def floor10(packs):
    return {g: (np.maximum(p, 0.10 * CAPACITY_KWH[g]), a) for g, (p, a) in packs.items()}


def main():
    base = floor10(load_pack("p8_base"))
    phys = floor10(load_pack("p8_phys"))

    s_base, s_phys = total_score(base, None), total_score(phys, None)
    print(f"fold24 점수:  base(v5) {s_base:.4f}   phys(v8) {s_phys:.4f}   차이 {s_phys - s_base:+.4f}")

    bb, bp = bootstrap_scores(base, phys)
    diff = bp - bb
    print(f"\n블록 부트스트랩 (7일 블록, {N_BOOT}회)")
    print(f"  단일 모델 점수 SE      : ±{bb.std():.4f}   95%CI [{np.percentile(bb,2.5):.4f}, {np.percentile(bb,97.5):.4f}]")
    print(f"  개선폭(phys-base) 평균 : {diff.mean():+.4f}  SE ±{diff.std():.4f}")
    print(f"  95%CI                  : [{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]")
    print(f"  P(개선>0)              : {(diff > 0).mean():.1%}")

    print("\n오차 구조 (v8, fold24, 평가 대상 시간대만)")
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        pred, act = phys[g]
        err, earn, mx, m = score_parts(pred, act, cap)
        band6 = (err <= 0.06).mean()
        w6 = (earn[err <= 0.06].sum() / mx.sum()) if mx.sum() else 0
        bias = ((pred[m] - act[m]) / cap).mean()
        print(f"  {g}: 평가시간 {m.sum():,}  NMAE {err.mean():.4f}  "
              f"6%밴드 적중 {band6:.1%}(발전량가중 {w6/1.0:.1%})  평균편향 {bias:+.3f}cap")

    # 실발전 구간별 오차 — 어디서 점수를 잃는가
    print("\n실발전 구간별 NMAE / 6%밴드 적중률 (3그룹 합산)")
    edges = [0.10, 0.25, 0.45, 0.70, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        errs, hits, wts = [], [], []
        for g in GROUPS:
            cap = CAPACITY_KWH[g]
            pred, act = phys[g]
            sel = (act >= lo * cap) & (act < hi * cap)
            if sel.sum() == 0:
                continue
            e = np.abs(pred[sel] - act[sel]) / cap
            errs.append(e)
            hits.append(e <= 0.06)
            wts.append(act[sel])
        e = np.concatenate(errs); h = np.concatenate(hits); w = np.concatenate(wts)
        share = w.sum()
        print(f"  실발전 {lo:.0%}~{hi:.0%}cap: n={len(e):,}  NMAE {e.mean():.4f}  "
              f"6%적중 {h.mean():.1%}  발전량비중 {share/sum(np.concatenate([phys[g][1] for g in GROUPS])):.1%}")


if __name__ == "__main__":
    main()
