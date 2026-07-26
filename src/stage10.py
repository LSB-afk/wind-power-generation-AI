"""v10 스크리닝 — 앙상블에 모델 계열 다양성 추가.

v9 는 서로 다른 **피처 표현** 3개를 평균해 Public +0.003 을 얻었고 로컬→Public
전이가 거의 1:1 이었다. 다음 축은 **모델 계열** 다양성이다.

후보 멤버 (모두 full 표현 위에서 학습):
  mlp  : 핀볼 손실 MLP (torch) — 트리가 아니라 부드럽게 외삽한다
  cat  : CatBoost Quantile — ordered boosting, LightGBM 과 분기 방식이 다르다
  steep: 급경사 구간(25~70%cap) 가중 LightGBM — 6%밴드 적중률이 가장 낮은 구간 표적

판정은 v9 와 동일한 게이트: 페어드 블록 부트스트랩(7일 x 2000) + 월 블록 부호검정.
기존 v9 팩(p9_base/full/fullterr)을 재사용하므로 새 멤버만 학습하면 된다.

사용법: python src/stage10.py mlp|cat|steep|report
"""
import sys

import numpy as np
import pandas as pd

from config import CAPACITY_KWH, GROUPS, ROOT
from stage9 import (CACHE, FILTER, FOLDS, G3, G12, load_feats, month_sign,
                    paired_boot, pooled_frame, score, solo_frame)

SEEDS = (42, 202, 777)
ALPHA = 0.60


# ─────────────────────── 멤버별 학습기 ───────────────────────

def fit_mlp(tr_X, tr_y, va_X, cap):
    """핀볼 손실 MLP. 표준화 + 중앙값 대치 후 학습한다 (NaN 을 못 다루므로)."""
    import torch
    import torch.nn as nn

    med = np.nanmedian(tr_X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda A: np.where(np.isfinite(A), A, med)
    Xtr, Xva = fill(tr_X), fill(va_X)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr = tr_y / cap

    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(Xva, dtype=torch.float32)
    preds = []
    for s in SEEDS:
        torch.manual_seed(s)
        net = nn.Sequential(
            nn.Linear(xt.shape[1], 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        n = len(xt)
        for epoch in range(60):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, 512):
                idx = perm[i:i + 512]
                opt.zero_grad()
                r = yt[idx] - net(xt[idx])
                # 핀볼 손실: alpha 분위 추정
                loss = torch.maximum(ALPHA * r, (ALPHA - 1) * r).mean()
                loss.backward()
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(xv).squeeze(1).numpy())
    return np.mean(preds, axis=0) * cap


def fit_cat(tr_X, tr_y, va_X, cap):
    from catboost import CatBoostRegressor
    preds = []
    for s in SEEDS:
        m = CatBoostRegressor(loss_function=f"Quantile:alpha={ALPHA}",
                              iterations=1500, learning_rate=0.05, depth=8,
                              rsm=0.3, l2_leaf_reg=3.0, random_seed=s,
                              verbose=False, allow_writing_files=False)
        m.fit(np.nan_to_num(tr_X, nan=-999.0), tr_y)
        preds.append(m.predict(np.nan_to_num(va_X, nan=-999.0)))
    return np.mean(preds, axis=0)


def fit_steep(tr_X, tr_y, va_X, cap):
    """급경사 구간 가중 LightGBM — 25~70%cap 표본에 2배 가중."""
    import lightgbm as lgb
    from stage9 import params
    r = tr_y / cap
    w = np.where((r >= 0.25) & (r <= 0.70), 2.0, 1.0)
    preds = []
    for s in SEEDS:
        p = dict(params(tr_X.shape[1]), seed=s)
        m = lgb.train(p, lgb.Dataset(tr_X, tr_y, weight=w), 1500)
        preds.append(m.predict(va_X))
    return np.mean(preds, axis=0)


FITTERS = {"mlp": fit_mlp, "cat": fit_cat, "steep": fit_steep}


def run(kind):
    fitter = FITTERS[kind]
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
                pr = fitter(ptr[pcols].to_numpy(dtype=np.float32),
                            ptr["y_norm"].to_numpy() * cap,
                            vx[pcols].to_numpy(dtype=np.float32), cap)
            else:
                tr = solo_frame(data, cols, g, pot, mism, tr_years)
                pr = fitter(tr[cols].to_numpy(dtype=np.float32),
                            tr["_y"].to_numpy(),
                            va[cols].to_numpy(dtype=np.float32), cap)
            out[f"val|{fold}|{g}"] = np.clip(pr, 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
            print(f"  [{kind}/{fold}/{g}] 완료", flush=True)
    np.savez(CACHE / f"p10_{kind}.npz", **out)
    print(f"저장: cache/p10_{kind}.npz")


# ─────────────────────── 판정 ───────────────────────

def blend(packs, names, weights=None):
    ref = packs[names[0]]
    w = np.array(weights if weights is not None else [1.0] * len(names), dtype=float)
    w = w / w.sum()
    out = {}
    for k in ref:
        out[k] = (sum(wi * packs[n][k] for wi, n in zip(w, names))
                  if k.startswith("val|") else ref[k])
    return out


def mix(packs, extra, weight):
    """v9 코어(동등가중 3표현)에 보조 멤버를 weight 비중으로 섞는다."""
    core = blend(packs, ["base", "full", "fullterr"])
    tmp = dict(packs)
    tmp["_core"] = core
    return blend(tmp, ["_core", extra], [1.0 - weight, weight])


def report():
    rng = np.random.default_rng(0)
    packs = {}
    for t in ("base", "full", "fullterr"):
        packs[t] = dict(np.load(CACHE / f"p9_{t}.npz", allow_pickle=True))
    for t in FITTERS:
        p = CACHE / f"p10_{t}.npz"
        if p.exists():
            packs[t] = dict(np.load(p, allow_pickle=True))

    v9 = ["base", "full", "fullterr"]
    avail = [t for t in FITTERS if t in packs]

    print("멤버 단독 점수")
    for t in v9 + avail:
        print(f"  {t:>10} fold24 {score(packs[t],'fold24',GROUPS):.4f}  "
              f"fold23 {score(packs[t],'fold23',G12):.4f}")

    print("\n보조 멤버 가중 탐색 (v9 코어에 섞는 비중)")
    built = {"v9(기준)": blend(packs, v9)}
    for t in avail:
        for w in (0.10, 0.20, 0.30):
            b = mix(packs, t, w)
            s24, s23 = score(b, "fold24", GROUPS), score(b, "fold23", G12)
            print(f"  v9+{t}@{w:.2f}: fold24 {s24:.4f}  fold23 {s23:.4f}")
            built[f"v9+{t}@{w:.2f}"] = b
    if len(avail) > 1:
        built["v9+" + "+".join(avail)] = blend(packs, v9 + avail,
                                               [1, 1, 1] + [0.5] * len(avail))

    ref = built["v9(기준)"]
    for n, b in built.items():
        if n == "v9(기준)":
            continue
        print(f"\n=== {n} vs v9 ===")
        ok = True
        for fold, groups in (("fold24", GROUPS), ("fold23", G12)):
            df = paired_boot(ref, b, fold, groups, rng)
            w, t = month_sign(ref, b, fold, groups)
            sig = df.mean() - 2 * df.std() > 0
            ok &= (df.mean() > 0)
            print(f"  {fold}: {df.mean():+.4f} ±{df.std():.4f} "
                  f"P(>0)={100*(df>0).mean():.1f}% 월별승 {w}/{t} "
                  f"{'2SE통과' if sig else '2SE미달'}")
        print(f"  → 양 폴드 양수: {'예' if ok else '아니오'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode in FITTERS:
        run(mode)
    else:
        report()
