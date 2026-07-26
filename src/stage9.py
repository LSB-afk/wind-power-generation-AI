"""v9 스크리닝 — 전처리 재설계(전 변수 x 전 격자) + 지형 보정.

v6/v8 의 교훈: 단일 폴드 절대점수 비교는 SE ±0.006 으로 무의미하다.
문헌 권고(블록 부트스트랩 + 월 블록 부호검정)에 따라 페어드 검정으로만 판정한다.

후보:
  A base   : v8 (348피처)
  B full   : 전 변수 x 전 격자 (1577피처) — 허브 외삽 기점 수정·Ri_b·가온도 밀도 포함
  C full+tc: B + 풍향 섹터 지형 보정 + SCADA 경험적 파워커브 피처

사용법: python src/stage9.py [A|B|C|report]
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, ROOT
from exp_runner import BASE_PARAMS, add_context, load_cache

CACHE = ROOT / "cache"
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"
SEEDS = (42, 202, 777)
FOLDS = {"fold24": ([2022, 2023], 2024, GROUPS), "fold23": ([2022], 2023, G12)}
FILTER, FLOOR = 0.05, 0.10
BLOCK, N_BOOT = 168, 2000


def params(n_feat):
    # 피처가 많아질수록 열 샘플링을 낮춰 트리 다양성을 유지한다
    ff = 0.7 if n_feat < 500 else (0.35 if n_feat < 1000 else 0.25)
    return dict(BASE_PARAMS, objective="quantile", alpha=0.60, feature_fraction=ff)


def load_feats(kind):
    """kind: 'base' | 'full'  → (피처 DataFrame, 라벨, potential, mismatch)"""
    labels = pd.read_parquet(CACHE / "train_labels.parquet")
    if kind == "base":
        base, ldaps_raw, _, _ = load_cache()
        davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
            "forecast_kst_dtm")["data_available_kst_dtm"]
        feat = add_context(base, davail).join(
            pd.read_parquet(CACHE / "train_phys.parquet"), how="left")
    else:
        feat = pd.read_parquet(CACHE / "full_train.parquet")
    pot = pd.read_parquet(CACHE / "scada_potential.parquet")
    mism = pd.read_parquet(CACHE / "scada_mismatch.parquet")
    return feat, labels, pot, mism


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


def fit(tr_X, tr_y, va_X, va_y, cat=None):
    preds = []
    for s in SEEDS:
        p = dict(params(tr_X.shape[1]), seed=s)
        dtr = lgb.Dataset(tr_X, tr_y, categorical_feature=cat or "auto")
        m = lgb.train(p, dtr, 5000,
                      valid_sets=[lgb.Dataset(va_X, va_y, reference=dtr)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(va_X, num_iteration=m.best_iteration))
    return np.mean(preds, axis=0)


def run(kind, with_terrain, tag):
    feat, labels, pot, mism = load_feats(kind)
    out = {}
    for fold, (tr_years, va_year, groups) in FOLDS.items():
        f = feat
        if with_terrain:
            from terrain import build_terrain_features
            f = feat.join(build_terrain_features(feat, tr_years), how="left")
        data = f.join(labels, how="inner")
        cols = list(f.columns)
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == va_year]
            if g == G3:
                ptr, pcols = pooled_frame(data, cols, pot, mism, tr_years)
                vx = va[cols].copy(); vx["gid"] = 3
                pr = fit(ptr[pcols], ptr["y_norm"], vx[pcols], va[g] / cap,
                         cat=["gid"]) * cap
            else:
                tr = solo_frame(data, cols, g, pot, mism, tr_years)
                pr = fit(tr[cols], tr["_y"], va[cols], va[g])
            out[f"val|{fold}|{g}"] = np.clip(pr, 0, cap)
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            out[f"idx|{fold}|{g}"] = va.index.month.to_numpy()
            print(f"  [{tag}/{fold}/{g}] 완료", flush=True)
    np.savez(CACHE / f"p9_{tag}.npz", **out)
    print(f"저장: cache/p9_{tag}.npz")


# ─────────────────────── 채점·검정 ───────────────────────

def _sc(pred, act, cap, sel=None):
    p = np.clip(np.maximum(pred, FLOOR * cap), 0, cap)
    m = act >= 0.10 * cap
    if sel is not None:
        m &= sel
    if m.sum() == 0:
        return np.nan, np.nan
    e = np.abs(p[m] - act[m]) / cap
    r = np.where(e <= 0.06, 4.0, np.where(e <= 0.08, 3.0, 0.0))
    return e.mean(), (r * act[m]).sum() / (4 * act[m]).sum()


def score(d, fold, groups, sel=None):
    nm, fi = [], []
    for g in groups:
        s = None if sel is None else sel[g]
        a, b = _sc(d[f"val|{fold}|{g}"], d[f"act|{fold}|{g}"], CAPACITY_KWH[g], s)
        nm.append(a); fi.append(b)
    return 0.5 * (1 - np.nanmean(nm)) + 0.5 * np.nanmean(fi)


def paired_boot(da, db, fold, groups, rng):
    n = len(da[f"act|{fold}|{groups[0]}"])
    diffs = []
    for _ in range(N_BOOT):
        st = rng.integers(0, max(n - BLOCK, 1), size=n // BLOCK + 1)
        i = np.clip(np.concatenate([np.arange(s, s + BLOCK) for s in st])[:n], 0, n - 1)
        sub = lambda d: {k: (v[i] if v.ndim == 1 and len(v) == n else v) for k, v in d.items()}
        diffs.append(score(sub(db), fold, groups) - score(sub(da), fold, groups))
    return np.array(diffs)


def month_sign(da, db, fold, groups):
    """월 블록별 개선 부호 — 폴드 내 준복제 표본으로 부호검정."""
    months = da[f"idx|{fold}|{groups[0]}"]
    wins = 0
    tot = 0
    for m in range(1, 13):
        sel = {g: da[f"idx|{fold}|{g}"] == m for g in groups}
        a, b = score(da, fold, groups, sel), score(db, fold, groups, sel)
        if np.isfinite(a) and np.isfinite(b):
            tot += 1
            wins += b > a
    return wins, tot


def report():
    rng = np.random.default_rng(0)
    packs = {}
    for t in ("base", "full", "fullterr"):
        p = CACHE / f"p9_{t}.npz"
        if p.exists():
            packs[t] = dict(np.load(p, allow_pickle=True))
    if "base" not in packs:
        print("base 팩이 없습니다. python src/stage9.py A 먼저 실행"); return

    print(f"{'후보':>10} {'fold24':>9} {'fold23':>9}")
    for t, d in packs.items():
        print(f"{t:>10} {score(d,'fold24',GROUPS):9.4f} {score(d,'fold23',G12):9.4f}")

    base = packs["base"]
    for t, d in packs.items():
        if t == "base":
            continue
        print(f"\n=== {t} vs base (페어드 블록 부트스트랩 7일블록 x {N_BOOT}) ===")
        ok = True
        for fold, groups in (("fold24", GROUPS), ("fold23", G12)):
            df = paired_boot(base, d, fold, groups, rng)
            w, n = month_sign(base, d, fold, groups)
            sig = df.mean() - 2 * df.std() > 0
            ok &= sig
            print(f"  {fold}: {df.mean():+.4f} ±{df.std():.4f}  "
                  f"95%CI[{np.percentile(df,2.5):+.4f},{np.percentile(df,97.5):+.4f}]  "
                  f"P(>0)={100*(df>0).mean():.1f}%  월별승 {w}/{n}  "
                  f"{'통과' if sig else '미달'}(평균>2SE)")
        print(f"  → 채택 {'가능' if ok else '불가'} (양 폴드 모두 평균>2SE 필요)")


def sweep():
    """분위 alpha x 학습필터 재조정.

    alpha=0.60, filter=0.05 는 v3 시절 낡은 파이프라인에서 정한 값이고
    potential 타깃·floor10 도입 이후 한 번도 재조정하지 않았다.
    평가가 '실발전>=10%cap' 조건부이므로 filter=0.10 이 그 조건부 분포와 정렬된다.
    """
    feat, labels, pot, mism = load_feats("base")
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    global FILTER
    rows = []
    for alpha in (0.55, 0.60, 0.65, 0.70):
        for filt in (0.05, 0.10):
            FILTER = filt
            sc = {}
            for fold, (tr_years, va_year, groups) in FOLDS.items():
                pk = {}
                for g in groups:
                    cap = CAPACITY_KWH[g]
                    va = data.dropna(subset=[g])
                    va = va[va.index.year == va_year]
                    tr = solo_frame(data, cols, g, pot, mism, tr_years)
                    p = dict(params(len(cols)), alpha=alpha, seed=42)
                    dtr = lgb.Dataset(tr[cols], tr["_y"])
                    m = lgb.train(p, dtr, 5000,
                                  valid_sets=[lgb.Dataset(va[cols], va[g], reference=dtr)],
                                  callbacks=[lgb.early_stopping(200, verbose=False)])
                    pk[f"val|{fold}|{g}"] = np.clip(
                        m.predict(va[cols], num_iteration=m.best_iteration), 0, cap)
                    pk[f"act|{fold}|{g}"] = va[g].to_numpy()
                # 그룹3은 여기서 제외하고 그룹1/2 부분점수로 비교 (상대 순위만 필요)
                sc[fold] = score(pk, fold, [g for g in groups if g != G3])
            rows.append((alpha, filt, sc["fold24"], sc["fold23"]))
            print(f"  alpha={alpha:.2f} filter={filt:.2f}  "
                  f"fold24 {sc['fold24']:.4f}  fold23 {sc['fold23']:.4f}", flush=True)
    df = pd.DataFrame(rows, columns=["alpha", "filter", "fold24", "fold23"])
    df["mean"] = df[["fold24", "fold23"]].mean(axis=1)
    print("\n양 폴드 평균 상위:")
    print(df.nlargest(4, "mean").to_string(index=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "A":
        run("base", False, "base")
    elif mode == "B":
        run("full", False, "full")
    elif mode == "C":
        run("full", True, "fullterr")
    elif mode == "sweep":
        sweep()
    else:
        report()
