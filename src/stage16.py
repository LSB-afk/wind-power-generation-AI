"""v16 스크리닝 — 재검증한 적 없는 근본 설계 선택들.

지금까지 손댄 적 없거나, v5 시절에 정한 뒤 파이프라인이 완전히 바뀌었는데도
그대로 유지된 것들을 다시 묻는다. 후처리(평활 + 하한 0.16)를 적용한 상태로
비교해 최종 파이프라인 기준의 이득을 본다.

  T0 현행       : potential 타깃 (v5 이후 고정)
  T1 실라벨      : 실제 KPX 발전량을 그대로 타깃으로
  T2 혼합타깃    : potential 과 실라벨의 평균
  T3 피처 가지치기: 중요도 상위 K개만 (1577개는 과다할 수 있다)
  T4 필터 완화   : 저발전 학습 제외 기준 5%cap → 0/10%

사용법: python src/stage16.py <T0|T1|T2|T3|T4> | report
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, ROOT
from exp_runner import BASE_PARAMS
from smooth import block_id, _denoise
from stage9 import CACHE, FOLDS, G3, G12, load_feats, month_sign, paired_boot

FLOOR = 0.16
SEEDS = (42, 202, 777)


def c7(seed):
    return dict(BASE_PARAMS, objective="quantile", alpha=0.60, seed=seed,
                learning_rate=0.02, num_leaves=63, min_data_in_leaf=100,
                feature_fraction=0.20, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=30.0)


def targets(data, pot, g, mode):
    """학습 타깃 선택. 검증·채점은 언제나 실제 라벨 기준이다."""
    raw = data[g]
    p = pot[f"{g}_potential"].reindex(data.index)
    if mode == "potential":
        return p
    if mode == "raw":
        return raw
    return (p + raw) / 2.0          # 혼합


def solo_frame(data, cols, g, pot, mism, years, mode, filt):
    cap = CAPACITY_KWH[g]
    d = data.dropna(subset=[g])
    tr = d[d.index.year.isin(years)].copy()
    tr["_y"] = targets(tr, pot, g, mode)
    tr = tr[~mism[f"{g}_mismatch"].reindex(tr.index).fillna(False)].dropna(subset=["_y"])
    return tr[tr["_y"] >= filt * cap] if filt > 0 else tr


def pooled_frame(data, cols, pot, mism, years, mode, filt):
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g]).copy()
        d["_y"] = targets(d, pot, g, mode)
        d = d[~mism[f"{g}_mismatch"].reindex(d.index).fillna(False)].dropna(subset=["_y"])
        f = d[cols].copy()
        f["y_norm"] = d["_y"] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        frames.append(f)
    p = pd.concat(frames)
    p = p[p.index.year.isin(years)]
    return (p[p["y_norm"] >= filt] if filt > 0 else p), cols + ["gid"]


def run(tag, mode="potential", filt=0.05, topk=None):
    feat, labels, pot, mism = load_feats("full")
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)

    if topk:                                  # 중요도 상위 K개로 가지치기
        d = data.dropna(subset=["kpx_group_1"])
        d = d[d.index.year <= 2023]
        d = d[d["kpx_group_1"] >= 0.05 * CAPACITY_KWH["kpx_group_1"]]
        m = lgb.train(c7(42), lgb.Dataset(d[cols], d["kpx_group_1"] / CAPACITY_KWH["kpx_group_1"]), 800)
        imp = pd.Series(m.feature_importance("gain"), index=cols)
        cols = list(imp.nlargest(topk).index)
        print(f"  피처 {len(feat.columns)} → {len(cols)}개로 가지치기", flush=True)

    out = {}
    for fold, (tr_years, va_year, groups) in FOLDS.items():
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == va_year]
            preds = []
            for s in SEEDS:
                if g == G3:
                    ptr, pcols = pooled_frame(data, cols, pot, mism, tr_years, mode, filt)
                    vx = va[cols].copy(); vx["gid"] = 3
                    dtr = lgb.Dataset(ptr[pcols], ptr["y_norm"], categorical_feature=["gid"])
                    m = lgb.train(c7(s), dtr, 6000,
                                  valid_sets=[lgb.Dataset(vx[pcols], va[g] / cap, reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    preds.append(m.predict(vx[pcols], num_iteration=m.best_iteration) * cap)
                else:
                    tr = solo_frame(data, cols, g, pot, mism, tr_years, mode, filt)
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
    np.savez(CACHE / f"p16_{tag}.npz", **out)
    print(f"저장: cache/p16_{tag}.npz")


# ─────────────────── 최종 파이프라인 기준 채점 ───────────────────

def finalize(pk):
    """평활 + 하한 0.16 적용 — 실제 제출과 같은 후처리."""
    out = dict(pk)
    for k in [k for k in pk if k.startswith("val|")]:
        _, fold, g = k.split("|")
        cap = CAPACITY_KWH[g]
        idx = pd.DatetimeIndex(pk[f"ts|{fold}|{g}"])
        pr = np.clip(np.maximum(pk[k], 0.10 * cap), 0, cap)
        pr = pd.Series(pr, index=idx).groupby(block_id(idx)).transform(_denoise).to_numpy()
        out[k] = np.clip(np.maximum(pr, FLOOR * cap), 0, cap)
    return out


def stats(pk, fold, gs):
    nm, fi = [], []
    for g in gs:
        cap = CAPACITY_KWH[g]
        pr, ac = pk[f"val|{fold}|{g}"], pk[f"act|{fold}|{g}"]
        m = ac >= 0.10 * cap
        e = np.abs(pr[m] - ac[m]) / cap
        r = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
        nm.append(e.mean()); fi.append((r * ac[m]).sum() / (4 * ac[m]).sum())
    return 0.5 * (1 - np.mean(nm)) + 0.5 * np.mean(fi), np.mean(fi)


def report():
    tags = [t for t in ("T0", "T1", "T2", "T3", "T4")
            if (CACHE / f"p16_{t}.npz").exists()]
    if "T0" not in tags:
        print("T0 기준선이 없습니다. python src/stage16.py T0 먼저 실행"); return
    P = {t: finalize(dict(np.load(CACHE / f"p16_{t}.npz", allow_pickle=True))) for t in tags}
    print(f"{'후보':>6} {'fold24':>9} {'FICR24':>8} {'fold23':>9} {'FICR23':>8}")
    for t in tags:
        a, fa = stats(P[t], "fold24", GROUPS)
        b, fb = stats(P[t], "fold23", G12)
        print(f"{t:>6} {a:9.4f} {fa:8.4f} {b:9.4f} {fb:8.4f}")
    rng = np.random.default_rng(0)
    for t in tags:
        if t == "T0":
            continue
        print(f"\n=== {t} vs T0 ===")
        ok = True
        for fold, gs in (("fold24", GROUPS), ("fold23", G12)):
            df = paired_boot(P["T0"], P[t], fold, gs, rng)
            w, n = month_sign(P["T0"], P[t], fold, gs)
            ok &= df.mean() > 0
            print(f"  {fold}: {df.mean():+.4f} ±{df.std():.4f} P(>0)={100*(df>0).mean():.1f}% "
                  f"월별승 {w}/{n} {'2SE통과' if df.mean()-2*df.std()>0 else '2SE미달'}")
        print(f"  → 양 폴드 양수: {'예' if ok else '아니오'}")


CONF = {"T0": dict(mode="potential", filt=0.05),
        "T1": dict(mode="raw", filt=0.05),
        "T2": dict(mode="mix", filt=0.05),
        "T3": dict(mode="potential", filt=0.05, topk=400),
        "T4": dict(mode="potential", filt=0.10)}

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "report"
    if m in CONF:
        run(m, **CONF[m])
    else:
        report()
