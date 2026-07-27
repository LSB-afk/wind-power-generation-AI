"""v13 스크리닝 — 다분위 예측분포 + 의사결정 이론 점예측.

점수식 분해상 밴드 적중 1회는 NMAE 로 환산해 `a_t/ā` 배(정격 근처면 2.0 x 설비용량)
가치가 있어 FICR 항이 의사결정을 지배한다. 그런데 우리는 quantile(0.60) 로
중앙값 근처를 예측해 왔다. 조건부 분포를 다분위로 세우고 기대 점수 최대점을 고른다.

v7 에서 같은 취지로 시도했다가 실패했는데, 그때는 분위가 7개뿐이라 폭 12%cap 창을
분해할 해상도가 없었고 후보 격자도 중앙값 주변으로 좁았다. 이번엔 분위 11개 +
전역 후보 격자 + 평가 필터(실발전 10%cap)를 적분 구간에 반영한다.

사용법: python src/stage13.py fit [base|full]   → 분위 예측 캐시
        python src/stage13.py report
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS
from decision import optimal_point
from exp_runner import BASE_PARAMS
from stage9 import (CACHE, FOLDS, G3, G12, load_feats, month_sign, paired_boot,
                    pooled_frame, score, solo_frame)

TAUS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.93, 0.98])
FLOOR = 0.10
SEED = 42
ROUNDS = 1500


def c7(alpha, seed):
    return dict(BASE_PARAMS, objective="quantile", alpha=alpha, seed=seed,
                learning_rate=0.02, num_leaves=63, min_data_in_leaf=100,
                feature_fraction=0.20, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=30.0)


def fit_quantiles(rep):
    feat, labels, pot, mism = load_feats(rep)
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    out = {}
    for fold, (tr_years, va_year, groups) in FOLDS.items():
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == va_year]
            if g == G3:
                ptr, pcols = pooled_frame(data, cols, pot, mism, tr_years)
                X, y, VX = ptr[pcols], ptr["y_norm"], va[cols].copy()
                VX["gid"] = 3
                VX, cat, sc = VX[pcols], ["gid"], cap
            else:
                tr = solo_frame(data, cols, g, pot, mism, tr_years)
                X, y, VX, cat, sc = tr[cols], tr["_y"] / cap, va[cols], None, cap
            Q = []
            for a in TAUS:
                m = lgb.train(c7(a, SEED),
                              lgb.Dataset(X, y, categorical_feature=cat or "auto"),
                              ROUNDS)
                Q.append(m.predict(VX) * sc)
            out[f"Q|{fold}|{g}"] = np.clip(np.stack(Q, axis=1), 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
            # 학습 연도의 평가 대상 평균 발전량 — 기대 점수 계산용 (정답 미사용)
            lab = data[g].dropna()
            lab = lab[lab.index.year.isin(tr_years)]
            out[f"abar|{fold}|{g}"] = np.array([lab[lab >= 0.10 * cap].mean()])
            print(f"  [{rep}/{fold}/{g}] 분위 {len(TAUS)}개 완료", flush=True)
    np.savez(CACHE / f"p13_{rep}.npz", **out)
    print(f"저장: cache/p13_{rep}.npz")


def build_packs(rep):
    d = dict(np.load(CACHE / f"p13_{rep}.npz", allow_pickle=True))
    med, opt = {}, {}
    for fold, (_, _, groups) in FOLDS.items():
        for g in groups:
            cap = CAPACITY_KWH[g]
            Q = d[f"Q|{fold}|{g}"]
            abar = float(d[f"abar|{fold}|{g}"][0])
            base = Q[:, int(np.argmin(np.abs(TAUS - 0.60)))]     # 기존 q60 상당
            for pk, val in ((med, base),
                            (opt, optimal_point(Q, TAUS, cap, abar))):
                pk[f"val|{fold}|{g}"] = np.clip(val, 0, cap)
                pk[f"act|{fold}|{g}"] = d[f"act|{fold}|{g}"]
                pk[f"idx|{fold}|{g}"] = d[f"idx|{fold}|{g}"]
    return med, opt


def detail(pk, fold, groups):
    nm, fi = [], []
    for g in groups:
        cap = CAPACITY_KWH[g]
        pr = np.clip(np.maximum(pk[f"val|{fold}|{g}"], FLOOR * cap), 0, cap)
        ac = pk[f"act|{fold}|{g}"]
        m = ac >= 0.10 * cap
        e = np.abs(pr[m] - ac[m]) / cap
        r = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        nm.append(e.mean()); fi.append((r * ac[m]).sum() / (4 * ac[m]).sum())
    return 1 - np.mean(nm), np.mean(fi)


def report():
    rep = sys.argv[2] if len(sys.argv) > 2 else "base"
    med, opt = build_packs(rep)
    print(f"[{rep}] {'후보':>14} {'fold24':>9} {'1-NMAE':>9} {'FICR':>9} | "
          f"{'fold23':>9} {'1-NMAE':>9} {'FICR':>9}")
    for n, pk in (("q60(기준)", med), ("기대점수 최적", opt)):
        n24, f24 = detail(pk, "fold24", GROUPS)
        n23, f23 = detail(pk, "fold23", G12)
        print(f"{'':>6} {n:>14} {score(pk,'fold24',GROUPS):9.4f} {n24:9.4f} {f24:9.4f} | "
              f"{score(pk,'fold23',G12):9.4f} {n23:9.4f} {f23:9.4f}")
    rng = np.random.default_rng(0)
    print("\n=== 기대점수 최적 vs q60 ===")
    for fold, gs in (("fold24", GROUPS), ("fold23", G12)):
        df = paired_boot(med, opt, fold, gs, rng)
        w, t = month_sign(med, opt, fold, gs)
        print(f"  {fold}: {df.mean():+.4f} ±{df.std():.4f} P(>0)={100*(df>0).mean():.1f}% "
              f"월별승 {w}/{t} {'2SE통과' if df.mean()-2*df.std()>0 else '2SE미달'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "fit":
        fit_quantiles(sys.argv[2] if len(sys.argv) > 2 else "base")
    else:
        report()
