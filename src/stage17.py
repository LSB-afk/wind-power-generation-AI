"""v17 — 외부 GEFS 상층풍 결합 검정.

오라클 실험이 지목한 병목(오차의 85% 가 풍속 예보 오차)을 직접 공략한다.
표본 검정에서 GEFS 850 hPa 가 제공 LDAPS 보다 SCADA 실측 허브풍속을 잘 설명했고,
선형 결합 시 풍속 MAE 가 20~22% 줄었다. 이제 실제 발전량 예측에서 확인한다.

  G0 현행(full 1577)
  G1 full + GEFS 피처
  G2 full + GEFS + 교차(제공 예보와의 불일치도)

판정은 기존 게이트 그대로: 이중 폴드 페어드 블록 부트스트랩 + 월 부호검정.
후처리(평활 + 하한 0.16)를 적용한 최종 파이프라인 기준으로 비교한다.

사용법: python src/stage17.py G0|G1|G2 | report
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

import gefs_feat
from config import CAPACITY_KWH, GROUPS
from exp_runner import BASE_PARAMS
from stage9 import CACHE, FOLDS, G3, G12, load_feats, month_sign, paired_boot
from stage16 import finalize, stats

SEEDS = (42, 202, 777)
FILTER = 0.05


def c7(seed):
    return dict(BASE_PARAMS, objective="quantile", alpha=0.60, seed=seed,
                learning_rate=0.02, num_leaves=63, min_data_in_leaf=100,
                feature_fraction=0.20, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=30.0)


def make_data(tag):
    feat, labels, pot, mism = load_feats("full")
    if tag != "G0":
        gf = gefs_feat.build(feat.index)
        if tag == "G2":
            gf = gefs_feat.add_cross(feat, gf)
        feat = feat.join(gf, how="left")
    return feat.join(labels, how="inner"), list(feat.columns), pot, mism


def solo_frame(data, cols, g, pot, mism, years):
    cap = CAPACITY_KWH[g]
    d = data.dropna(subset=[g])
    tr = d[d.index.year.isin(years)].copy()
    tr["_y"] = pot[f"{g}_potential"].reindex(tr.index)
    tr = tr[~mism[f"{g}_mismatch"].reindex(tr.index).fillna(False)].dropna(subset=["_y"])
    return tr[tr["_y"] >= FILTER * cap]


def pooled_frame(data, cols, pot, mism, years):
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g]).copy()
        d["_y"] = pot[f"{g}_potential"].reindex(d.index)
        d = d[~mism[f"{g}_mismatch"].reindex(d.index).fillna(False)].dropna(subset=["_y"])
        f = d[cols].copy()
        f["y_norm"] = d["_y"] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        frames.append(f)
    p = pd.concat(frames)
    p = p[p.index.year.isin(years)]
    return p[p["y_norm"] >= FILTER], cols + ["gid"]


def run(tag):
    data, cols, pot, mism = make_data(tag)
    print(f"  피처 {len(cols)}개", flush=True)
    out = {}
    for fold, (tr_years, va_year, groups) in FOLDS.items():
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == va_year]
            preds = []
            for s in SEEDS:
                if g == G3:
                    ptr, pcols = pooled_frame(data, cols, pot, mism, tr_years)
                    vx = va[cols].copy(); vx["gid"] = 3
                    dtr = lgb.Dataset(ptr[pcols], ptr["y_norm"], categorical_feature=["gid"])
                    m = lgb.train(c7(s), dtr, 6000,
                                  valid_sets=[lgb.Dataset(vx[pcols], va[g] / cap, reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    preds.append(m.predict(vx[pcols], num_iteration=m.best_iteration) * cap)
                else:
                    tr = solo_frame(data, cols, g, pot, mism, tr_years)
                    dtr = lgb.Dataset(tr[cols], tr["_y"])
                    m = lgb.train(c7(s), dtr, 6000,
                                  valid_sets=[lgb.Dataset(va[cols], va[g], reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    preds.append(m.predict(va[cols], num_iteration=m.best_iteration))
            out[f"val|{fold}|{g}"] = np.clip(np.mean(preds, axis=0), 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
            out[f"ts|{fold}|{g}"] = va.index.values
            print(f"  [{tag}/{fold}/{g}] 완료", flush=True)
    np.savez(CACHE / f"p17_{tag}.npz", **out)
    print(f"저장: cache/p17_{tag}.npz")


def report():
    tags = [t for t in ("G0", "G1", "G2") if (CACHE / f"p17_{t}.npz").exists()]
    if "G0" not in tags:
        print("G0 기준선이 없습니다."); return
    P = {t: finalize(dict(np.load(CACHE / f"p17_{t}.npz", allow_pickle=True))) for t in tags}
    print(f"{'후보':>6} {'fold24':>9} {'FICR24':>8} {'fold23':>9} {'FICR23':>8}")
    for t in tags:
        a, fa = stats(P[t], "fold24", GROUPS)
        b, fb = stats(P[t], "fold23", G12)
        print(f"{t:>6} {a:9.4f} {fa:8.4f} {b:9.4f} {fb:8.4f}")
    rng = np.random.default_rng(0)
    for t in tags:
        if t == "G0":
            continue
        print(f"\n=== {t} vs G0 ===")
        ok = True; W = T = 0
        for fold, gs in (("fold24", GROUPS), ("fold23", G12)):
            df = paired_boot(P["G0"], P[t], fold, gs, rng)
            w, n = month_sign(P["G0"], P[t], fold, gs)
            ok &= df.mean() - 2 * df.std() > 0
            W += w; T += n
            print(f"  {fold}: {df.mean():+.4f} ±{df.std():.4f} P(>0)={100*(df>0).mean():.1f}% "
                  f"월별승 {w}/{n} {'2SE통과' if df.mean()-2*df.std()>0 else '2SE미달'}")
        print(f"  → 양 폴드 2SE {'통과' if ok else '미달'}, 월별 {W}/{T}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "report"
    if m in ("G0", "G1", "G2"):
        run(m)
    else:
        report()
