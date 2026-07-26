"""v11 스크리닝 — 밴드 포화 손실(FICR 직접 겨냥).

근거: Public 격차의 84% 가 FICR. 잔차를 줄여 1등과 같은 정확도(1-NMAE 0.879)에
맞춰도 우리 FICR 은 0.414 로 1등 0.465 에 못 미친다 → 오차 분포의 모양 문제.
pinball 대신 포화 손실을 섞어 "밴드 밖 가망 없는 시간대를 포기하고 밴드 근처를
안으로 밀어 넣도록" 학습시킨다.

사용법: python src/stage11.py sweep [seeds]   → w_band 격자 탐색 (이중 폴드)
        python src/stage11.py confirm <w>     → 승자를 3시드로 재확인
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from bandloss import make_objective
from config import CAPACITY_KWH, GROUPS
from stage9 import (CACHE, FOLDS, G3, G12, load_feats, month_sign, paired_boot,
                    params, pooled_frame, score, solo_frame)

FLOOR = 0.10


BASE_ROUNDS, REFINE_ROUNDS = 1200, 600


def fit_band(tr_X, tr_y_norm, va_X, w_band, seeds, cat=None):
    """2단계 학습: pinball 로 밴드 근처까지 데려간 뒤, 포화 손실로 다듬는다.

    포화 손실은 잔차가 밴드보다 훨씬 크면 기울기가 사라져 처음부터는 학습이 안 된다.
    그래서 1단계 예측을 init_score 로 주고 2단계에서 밴드 안으로 밀어 넣는다.
    w_band=0 이면 1단계만 수행해 기존 v9 레시피와 동일하다.
    """
    cf = list(cat) if cat else "auto"
    preds = []
    for s in seeds:
        p = dict(params(tr_X.shape[1]), seed=s)
        base = lgb.train(p, lgb.Dataset(tr_X, tr_y_norm, categorical_feature=cf),
                         BASE_ROUNDS)
        init_tr, init_va = base.predict(tr_X), base.predict(va_X)
        if w_band <= 0:
            preds.append(init_va)
            continue
        p2 = dict(params(tr_X.shape[1]), seed=s + 1)
        p2.pop("alpha", None)
        p2["objective"] = make_objective(w_band)
        p2["learning_rate"] = 0.01           # 다듬기 단계는 보수적으로
        dtr2 = lgb.Dataset(tr_X, tr_y_norm, init_score=init_tr, categorical_feature=cf)
        ref = lgb.train(p2, dtr2, REFINE_ROUNDS)
        preds.append(init_va + ref.predict(va_X))
    return np.mean(preds, axis=0)


def run(w_band, seeds, tag):
    feat, labels, pot, mism = load_feats("full")
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
                vx = va[cols].copy(); vx["gid"] = 3
                pr = fit_band(ptr[pcols], ptr["y_norm"], vx[pcols], w_band, seeds,
                              cat=["gid"]) * cap
            else:
                tr = solo_frame(data, cols, g, pot, mism, tr_years)
                pr = fit_band(tr[cols], tr["_y"] / cap, va[cols], w_band, seeds) * cap
            out[f"val|{fold}|{g}"] = np.clip(pr, 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
            print(f"  [w={w_band}/{fold}/{g}] 완료", flush=True)
    np.savez(CACHE / f"p11_{tag}.npz", **out)
    return out


def detail(d, fold, groups):
    """(1-NMAE, FICR) 분해 — 밴드 손실이 어느 항을 올리는지 보려고."""
    nm, fi = [], []
    for g in groups:
        cap = CAPACITY_KWH[g]
        pr = np.clip(np.maximum(d[f"val|{fold}|{g}"], FLOOR * cap), 0, cap)
        ac = d[f"act|{fold}|{g}"]
        m = ac >= 0.10 * cap
        e = np.abs(pr[m] - ac[m]) / cap
        r = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        nm.append(e.mean()); fi.append((r * ac[m]).sum() / (4 * ac[m]).sum())
    return 1 - np.mean(nm), np.mean(fi)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        seeds = (42, 202, 777)[:nseed]
        print(f"w_band 격자 탐색 ({nseed}시드)")
        print(f"{'w':>5} {'fold24':>9} {'1-NMAE':>9} {'FICR':>9} | {'fold23':>9} {'1-NMAE':>9} {'FICR':>9}")
        for w in (0.0, 0.3, 0.5, 0.7):
            d = run(w, seeds, f"w{int(w*10)}")
            n24, f24 = detail(d, "fold24", GROUPS)
            n23, f23 = detail(d, "fold23", G12)
            print(f"{w:5.1f} {score(d,'fold24',GROUPS):9.4f} {n24:9.4f} {f24:9.4f} | "
                  f"{score(d,'fold23',G12):9.4f} {n23:9.4f} {f23:9.4f}", flush=True)
    else:
        w = float(sys.argv[2])
        d = run(w, (42, 202, 777), f"confirm{int(w*10)}")
        n24, f24 = detail(d, "fold24", GROUPS)
        n23, f23 = detail(d, "fold23", G12)
        print(f"\n[3시드 w={w}] fold24 {score(d,'fold24',GROUPS):.4f} "
              f"(1-NMAE {n24:.4f}, FICR {f24:.4f}) | "
              f"fold23 {score(d,'fold23',G12):.4f} (1-NMAE {n23:.4f}, FICR {f23:.4f})")


if __name__ == "__main__":
    main()
