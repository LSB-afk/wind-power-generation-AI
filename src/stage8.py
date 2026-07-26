"""v8 후보 스크리닝 — v5 베이스, 이중 폴드 x 3시드.

채택 게이트 (v6/v7 실패 교훈): 이중 폴드(22→23, 22-23→24) 모두에서
3시드 평균 +0.003 이상. 단일 시드·단일 폴드 이득은 채택 근거가 아니다.

후보:
  E1 phys  : 물리 피처 팩 (허브고도 전단지수·대기 안정도·상층풍·환기율)
  E2 calib : FICR 기대효용 캘리브레이션 — 학습연도 OOF 잔차로만 오프셋 추정
  E3 cat   : CatBoost 이종 앙상블 블렌드

사용법: python src/stage8.py [base|phys|cat|report]
예측은 cache/p8_*.npz 에 캐시되어 재실행이 싸다.
"""
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, ROOT
from exp_runner import BASE_PARAMS, CACHE, add_context, group_score, load_cache
from metrics import competition_score
from postprocess import apply_post
from stage7 import pooled_frame, solo_frame

G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"
SEEDS3 = (42, 202, 777)
FOLDS = {"fold24": ([2022, 2023], 2024, GROUPS), "fold23": ([2022], 2023, G12)}
Q60 = dict(BASE_PARAMS, objective="quantile", alpha=0.60)
RATE_TIERS = ((0.06, 1.0), (0.08, 0.75))


# ──────────────────────── 데이터 ────────────────────────

def setup(with_phys: bool):
    base_feat, ldaps_raw, gfs_raw, labels = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    feat = add_context(base_feat, davail)
    if with_phys:
        phys = pd.read_parquet(CACHE / "train_phys.parquet")
        feat = feat.join(phys.reindex(feat.index), how="left")
    data = feat.join(labels, how="inner")
    pot = pd.read_parquet(CACHE / "scada_potential.parquet")
    mism = pd.read_parquet(CACHE / "scada_mismatch.parquet")
    return data, list(feat.columns), pot, mism


def oof_blocks(index, train_years):
    """학습연도 내 OOF 블록 — 연도가 2개 이상이면 연도 교차, 아니면 월 홀짝."""
    rows = index[index.year.isin(train_years)]
    if len(train_years) >= 2:
        blocks = [rows[rows.year == y] for y in train_years]
    else:
        blocks = [rows[rows.month % 2 == 0], rows[rows.month % 2 == 1]]
    return [b for b in blocks if len(b) > 0]


# ──────────────────────── 모델 ────────────────────────

def _lgb_fit(tr_X, tr_y, va_X, va_y, seed, rounds=None, categorical=None):
    p = dict(Q60, seed=seed)
    dtrain = lgb.Dataset(tr_X, tr_y, categorical_feature=categorical or "auto")
    if rounds is not None:
        m = lgb.train(p, dtrain, rounds)
        return m.predict(va_X), rounds
    m = lgb.train(p, dtrain, 5000, valid_sets=[lgb.Dataset(va_X, va_y, reference=dtrain)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(va_X, num_iteration=m.best_iteration), m.best_iteration


def _cat_fit(tr_X, tr_y, va_X, va_y, seed):
    from catboost import CatBoostRegressor, Pool
    m = CatBoostRegressor(loss_function="Quantile:alpha=0.6", iterations=4000,
                          learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                          random_seed=seed, verbose=0, allow_writing_files=False,
                          od_type="Iter", od_wait=200)
    m.fit(Pool(tr_X, tr_y), eval_set=Pool(va_X, va_y), use_best_model=True)
    return m.predict(va_X)


def predict_pack(tag, with_phys, model="lgb", seeds=SEEDS3, with_oof=True):
    """폴드 x 그룹별 (검증 예측, 학습연도 OOF 예측) 시드 평균을 캐시."""
    path = CACHE / f"p8_{tag}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    data, cols, pot, mism = setup(with_phys)
    print(f"[{tag}] 데이터 {data.shape[0]:,}행, 피처 {len(cols)}개")
    out = {}
    for fold, (train_years, valid_year, groups) in FOLDS.items():
        for g in groups:
            t0 = time.time()
            cap = CAPACITY_KWH[g]
            d = data.dropna(subset=[g])
            va = d[d.index.year == valid_year]
            pooled = g == G3
            if pooled:
                ptr, pcols = pooled_frame(data, cols, pot, mism, train_years)
                va_X = va[cols].copy()
                va_X["gid"] = 3
                va_X = va_X[pcols]
                tr_X, tr_y, kw = ptr[pcols], ptr["y_norm"], {"categorical": ["gid"]}
                scale, va_y = cap, va[g] / cap
            else:
                tr = solo_frame(data, cols, g, pot, mism, train_years)
                va_X, tr_X, tr_y, kw = va[cols], tr[cols], tr["_y"], {}
                scale, va_y = 1.0, va[g]

            preds, iters = [], []
            for s in seeds:
                if model == "cat":
                    preds.append(_cat_fit(tr_X, tr_y, va_X, va_y, s))
                    iters.append(0)
                else:
                    p, it = _lgb_fit(tr_X, tr_y, va_X, va_y, s, **kw)
                    preds.append(p)
                    iters.append(it)
            out[f"val|{fold}|{g}"] = np.mean(preds, axis=0) * scale
            out[f"act|{fold}|{g}"] = va[g].to_numpy()
            n_rounds = max(int(np.mean(iters)), 100)

            # OOF (학습연도만) — 캘리브레이션용. 검증연도 정보 미사용.
            if model == "lgb" and with_oof:
                oof = pd.Series(np.nan, index=d.index[d.index.year.isin(train_years)])
                for blk in oof_blocks(d.index, train_years):
                    if pooled:
                        sub = ptr[~ptr.index.isin(blk)]
                        bx = data.loc[blk, cols].copy()
                        bx["gid"] = 3
                        bx = bx[pcols]
                        p, _ = _lgb_fit(sub[pcols], sub["y_norm"], bx, None, seeds[0],
                                        rounds=n_rounds, categorical=["gid"])
                    else:
                        sub = tr[~tr.index.isin(blk)]
                        p, _ = _lgb_fit(sub[cols], sub["_y"], data.loc[blk, cols], None,
                                        seeds[0], rounds=n_rounds)
                    oof.loc[blk] = p * scale
                out[f"oof|{fold}|{g}"] = oof.to_numpy()
                out[f"oofact|{fold}|{g}"] = d.loc[oof.index, g].to_numpy()
            print(f"  [{fold}/{g}] rounds~{n_rounds} {time.time()-t0:.0f}s")
    np.savez(path, **out)
    return out


# ──────────────────────── E2: FICR 기대효용 캘리브레이션 ────────────────────────

def rate01(e):
    r = np.zeros_like(e)
    for thr, val in reversed(RATE_TIERS):
        r = np.where(e <= thr, val, r)
    return r


def _utility(p, a, cap, abar):
    """총점의 시간대별 기여 (평가 필터 밖은 0). 상수배 무시."""
    e = np.abs(np.clip(p, 0, cap) - a) / cap
    return np.where(a >= 0.10 * cap, -e + (a / abar) * rate01(e), 0.0)


def fit_offsets(p_oof, a_oof, cap, n_bins, dmax=0.20, min_rows=300):
    """OOF 예측 구간별 기대효용 최대 오프셋. 반환 (bin 중앙값, delta)."""
    ok = np.isfinite(p_oof) & np.isfinite(a_oof)
    p_oof, a_oof = p_oof[ok], a_oof[ok]
    ev = a_oof >= 0.10 * cap
    if ev.sum() == 0:
        return np.array([0.0]), np.array([0.0])
    abar = a_oof[ev].mean()
    grid = np.arange(-dmax, dmax + 1e-9, 0.0025) * cap
    while n_bins > 1 and len(p_oof) / n_bins < min_rows:
        n_bins -= 1
    edges = np.quantile(p_oof, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    centers, deltas = [], []
    for b in range(n_bins):
        sel = (p_oof >= edges[b]) & (p_oof < edges[b + 1])
        if sel.sum() < 30:
            continue
        p, a = p_oof[sel], a_oof[sel]
        u = np.array([_utility(p + d, a, cap, abar).mean() for d in grid])
        centers.append(np.median(p))
        deltas.append(grid[int(np.argmax(u))])
    if not centers:
        return np.array([0.0]), np.array([0.0])
    return np.array(centers), np.array(deltas)


def apply_offsets(pred, cap, centers, deltas):
    d = np.interp(pred, centers, deltas) if len(centers) > 1 else np.full_like(pred, deltas[0])
    return np.clip(pred + d, 0, cap)


def floor10(pred, cap):
    return apply_post(pred, cap, 1.0, 0.10 * cap)


# ──────────────────────── 채점 ────────────────────────

def score_fold(pack, fold, groups, transform=None):
    preds, acts = {}, {}
    for g in groups:
        cap = CAPACITY_KWH[g]
        p = np.clip(pack[f"val|{fold}|{g}"], 0, cap)
        if transform is not None:
            p = transform(p, g, fold)
        preds[g], acts[g] = floor10(p, cap), pack[f"act|{fold}|{g}"]
    if len(groups) == 3:
        r = competition_score(preds, acts)
        s = r["score"]
    else:
        s = float(np.mean([group_score(preds[g], acts[g], CAPACITY_KWH[g]) for g in groups]))
    per_g = {g: group_score(preds[g], acts[g], CAPACITY_KWH[g]) for g in groups}
    return s, per_g


def report(name, pack, transform=None, base=None):
    line = {}
    for fold, (_, _, groups) in FOLDS.items():
        s, per_g = score_fold(pack, fold, groups, transform)
        line[fold] = (s, per_g)
    msg = f"[{name}]"
    for fold in FOLDS:
        s, per_g = line[fold]
        d = f" ({s - base[fold][0]:+.4f})" if base else ""
        gs = " ".join(f"g{g[-1]}={per_g[g]:.4f}" for g in per_g)
        msg += f"\n  {fold} {s:.6f}{d}  {gs}"
        if base:
            msg += "  " + " ".join(f"g{g[-1]}{per_g[g]-base[fold][1][g]:+.4f}" for g in per_g)
    print(msg)
    return line


def calib_transform(pack, n_bins):
    cal = {}
    for fold, (_, _, groups) in FOLDS.items():
        for g in groups:
            key = f"oof|{fold}|{g}"
            if key not in pack:
                cal[(fold, g)] = (np.array([0.0]), np.array([0.0]))
                continue
            cal[(fold, g)] = fit_offsets(pack[key], pack[f"oofact|{fold}|{g}"],
                                         CAPACITY_KWH[g], n_bins)
    def t(p, g, fold):
        c, d = cal[(fold, g)]
        return apply_offsets(p, CAPACITY_KWH[g], c, d)
    return t, cal


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    packs = {}
    if mode in ("all", "base", "report"):
        packs["base"] = predict_pack("base", with_phys=False)
    if mode in ("all", "phys", "report"):
        packs["phys"] = predict_pack("phys", with_phys=True)
    if mode in ("all", "cat", "report"):
        packs["cat"] = predict_pack("cat", with_phys=False, model="cat")
    if mode not in ("all", "report"):
        return

    print("\n===== v8 스크리닝 (3시드, 이중 폴드) =====")
    base = report("B0_v5기준선", packs["base"])

    if "phys" in packs:
        report("E1_물리피처팩", packs["phys"], base=base)
        mix = {k: v for k, v in packs["base"].items()}
        for k in list(mix):
            if k.startswith("val|"):
                mix[k] = 0.5 * packs["base"][k] + 0.5 * packs["phys"][k]
            if k.startswith("oof|"):
                mix[k] = 0.5 * packs["base"][k] + 0.5 * packs["phys"][k]
        report("E1b_base+phys블렌드", mix, base=base)

    for nb in (1, 4, 8):
        t, cal = calib_transform(packs["base"], nb)
        report(f"E2_기대효용캘리브(bins={nb})", packs["base"], transform=t, base=base)
        for (fold, g), (c, d) in sorted(cal.items()):
            print(f"     {fold}/{g}: delta/cap = "
                  + ", ".join(f"{x/CAPACITY_KWH[g]:+.3f}" for x in d))

    if "cat" in packs:
        report("E3_CatBoost단독", packs["cat"], base=base)
        mix = dict(packs["base"])
        for k in list(mix):
            if k.startswith("val|"):
                mix[k] = 0.5 * packs["base"][k] + 0.5 * packs["cat"][k]
        report("E3b_LGBM+CatBoost블렌드", mix, base=base)


if __name__ == "__main__":
    main()
