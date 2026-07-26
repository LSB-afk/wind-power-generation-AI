"""v12 — LightGBM 하이퍼파라미터 탐색 (v1 이후 최초).

v1(195피처) 시절 정한 lr 0.03 / num_leaves 63 / feature_fraction 0.7 을
피처가 1577개로 8배 늘어난 지금까지 그대로 쓰고 있었다. 피처 규모가 바뀌면
최적 열 샘플링·트리 크기·정칙화도 바뀌는 게 정상이라 재탐색한다.

판정은 기존 게이트와 동일: 이중 폴드 동시 개선 + 페어드 부트스트랩.
탐색은 1시드로 하되 승자는 반드시 3시드로 재확인한다(v7-B 교훈).

사용법: python src/stage12.py search | confirm <idx>
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS
from exp_runner import BASE_PARAMS
from stage9 import (CACHE, FOLDS, G3, G12, load_feats, pooled_frame, score,
                    solo_frame)

FLOOR = 0.10

# 1577피처 x 1.8만행 환경에 맞춘 후보들. 현행(C0)을 기준으로 축을 하나씩 움직인다.
CONFIGS = [
    ("C0 현행",       dict(learning_rate=0.03, num_leaves=63,  min_data_in_leaf=40,
                          feature_fraction=0.25, bagging_fraction=0.8, lambda_l2=1.0)),
    ("C1 열샘플↑",    dict(learning_rate=0.03, num_leaves=63,  min_data_in_leaf=40,
                          feature_fraction=0.50, bagging_fraction=0.8, lambda_l2=1.0)),
    ("C2 열샘플↓",    dict(learning_rate=0.03, num_leaves=63,  min_data_in_leaf=40,
                          feature_fraction=0.12, bagging_fraction=0.8, lambda_l2=1.0)),
    ("C3 얕게+규제",  dict(learning_rate=0.03, num_leaves=31,  min_data_in_leaf=80,
                          feature_fraction=0.25, bagging_fraction=0.8, lambda_l2=10.0)),
    ("C4 깊게",       dict(learning_rate=0.03, num_leaves=127, min_data_in_leaf=20,
                          feature_fraction=0.25, bagging_fraction=0.8, lambda_l2=5.0)),
    ("C5 저lr+장기",  dict(learning_rate=0.012, num_leaves=63, min_data_in_leaf=40,
                          feature_fraction=0.25, bagging_fraction=0.8, lambda_l2=1.0)),
    ("C6 저lr+얕게",  dict(learning_rate=0.012, num_leaves=31, min_data_in_leaf=60,
                          feature_fraction=0.30, bagging_fraction=0.7, lambda_l2=5.0)),
    ("C7 강규제",     dict(learning_rate=0.02, num_leaves=63,  min_data_in_leaf=100,
                          feature_fraction=0.20, bagging_fraction=0.7, lambda_l2=30.0)),
]


def build(cfg, seed):
    return dict(BASE_PARAMS, objective="quantile", alpha=0.60, seed=seed,
                bagging_freq=1, verbosity=-1, **cfg)


def run_cfg(cfg, seeds, data, cols, pot, mism):
    out = {}
    for fold, (tr_years, va_year, groups) in FOLDS.items():
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == va_year]
            preds = []
            for s in seeds:
                p = build(cfg, s)
                if g == G3:
                    ptr, pcols = pooled_frame(data, cols, pot, mism, tr_years)
                    vx = va[cols].copy(); vx["gid"] = 3
                    dtr = lgb.Dataset(ptr[pcols], ptr["y_norm"], categorical_feature=["gid"])
                    m = lgb.train(p, dtr, 6000,
                                  valid_sets=[lgb.Dataset(vx[pcols], va[g] / cap, reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    preds.append(m.predict(vx[pcols], num_iteration=m.best_iteration) * cap)
                else:
                    tr = solo_frame(data, cols, g, pot, mism, tr_years)
                    dtr = lgb.Dataset(tr[cols], tr["_y"])
                    m = lgb.train(p, dtr, 6000,
                                  valid_sets=[lgb.Dataset(va[cols], va[g], reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    preds.append(m.predict(va[cols], num_iteration=m.best_iteration))
            out[f"val|{fold}|{g}"] = np.clip(np.mean(preds, axis=0), 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "search"
    feat, labels, pot, mism = load_feats("full")
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)

    if mode == "search":
        print(f"{'설정':>14} {'fold24':>9} {'fold23':>9} {'평균':>9}")
        rows = []
        for i, (name, cfg) in enumerate(CONFIGS):
            d = run_cfg(cfg, (42,), data, cols, pot, mism)
            s24, s23 = score(d, "fold24", GROUPS), score(d, "fold23", G12)
            np.savez(CACHE / f"p12_c{i}.npz", **d)
            rows.append((name, s24, s23))
            print(f"{name:>14} {s24:9.4f} {s23:9.4f} {(s24+s23)/2:9.4f}", flush=True)
        best = max(rows, key=lambda r: (r[1] + r[2]) / 2)
        print(f"\n최고: {best[0]}  (fold24 {best[1]:.4f}, fold23 {best[2]:.4f})")
    else:
        i = int(sys.argv[2])
        name, cfg = CONFIGS[i]
        d = run_cfg(cfg, (42, 202, 777), data, cols, pot, mism)
        np.savez(CACHE / f"p12_confirm{i}.npz", **d)
        print(f"[3시드 {name}] fold24 {score(d,'fold24',GROUPS):.4f}  "
              f"fold23 {score(d,'fold23',G12):.4f}")


if __name__ == "__main__":
    main()
