"""v7 실험: (A) 다중 분위 기대효용 최적 포인트, (B) NWP→SCADA 풍속 보정 피처.

사용법: python src/stage7.py

원칙 (v6 실패 교훈):
- 이중 폴드(22→23, 22-23→24) 모두 +0.003 이상 개선해야 채택.
  (v6의 +0.001~0.002 미세 이득은 Public에서 -0.003으로 역전됐음)
- 베이스는 검증·제출 이력이 가장 신뢰되는 v5 레시피
  (potential 타깃, q60/filter05, 그룹1/2 solo + 그룹3 pooled, floor10).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from exp_runner import BASE_PARAMS, CACHE, add_context, group_score, load_cache, score_str
from config import CAPACITY_KWH, GROUPS, RANDOM_SEED
from postprocess import apply_post
from scada import _GROUP_TURBINES, _load_scada

G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"
QUANTS = (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)
FILTER = 0.05
RATE_TIERS = ((0.06, 1.0), (0.08, 0.75))   # 오차율 임계, rate/4


# ──────────────────────── 공통 준비 ────────────────────────

def setup():
    base_feat, ldaps_raw, gfs_raw, labels = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    feat = add_context(base_feat, davail)
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    pot = pd.read_parquet(CACHE / "scada_potential.parquet")
    mism = pd.read_parquet(CACHE / "scada_mismatch.parquet")
    return data, cols, pot, mism


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
    pooled = pd.concat(frames)
    pooled = pooled[pooled.index.year.isin(years)]
    return pooled[pooled["y_norm"] >= FILTER], cols + ["gid"]


def fit_one(tr_X, tr_y, va_X, va_y, alpha, categorical=None, feature_cols=None):
    p = dict(BASE_PARAMS, objective="quantile", alpha=alpha, seed=RANDOM_SEED)
    dtrain = lgb.Dataset(tr_X, tr_y, categorical_feature=categorical or "auto")
    m = lgb.train(p, dtrain, 5000,
                  valid_sets=[lgb.Dataset(va_X, va_y, reference=dtrain)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(va_X, num_iteration=m.best_iteration)


def valid_rows(data, g, valid_year):
    d = data.dropna(subset=[g])
    return d[d.index.year == valid_year]


def floor10(pred, cap):
    return apply_post(pred, cap, 1.0, 0.10 * cap)


# ──────────────────────── A) 기대효용 최적 포인트 ────────────────────────

def rate_of(err_ratio):
    r = np.zeros_like(err_ratio)
    for thr, val in reversed(RATE_TIERS):
        r = np.where(err_ratio <= thr, val, r)
    return r


def utility_points(Q, cap, abar):
    """Q: (n, k) 분위 샘플. 시간대별 기대효용 최대 포인트 반환.

    u(p, a) = 1[a >= 0.1cap] * ( -|p-a|/cap + rate(|p-a|/cap) * a / (4/4 * abar) )
    (총점의 그룹 기여를 시간대별로 분해한 형태 — 상수 배 무시)
    """
    cand = np.linspace(0.0, cap, 241)                    # (c,)
    p_star = np.empty(len(Q))
    for lo in range(0, len(Q), 512):
        q = Q[lo:lo + 512]                               # (b, k)
        mask = (q >= 0.10 * cap)                         # (b, k)
        err = np.abs(cand[None, :, None] - q[:, None, :]) / cap   # (b, c, k)
        u = (-err + rate_of(err) * q[:, None, :] / abar) * mask[:, None, :]
        p_star[lo:lo + 512] = cand[np.argmax(u.mean(axis=2), axis=1)]
    return p_star


def run_A(data, cols, pot, mism):
    print("=== A) 다중 분위 기대효용 최적 포인트 ===")
    results = {}
    for fold, (train_years, valid_year, groups) in {
            "fold24": ([2022, 2023], 2024, GROUPS),
            "fold23": ([2022], 2023, G12)}.items():
        preds_med, preds_util, actuals = {}, {}, {}
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = valid_rows(data, g, valid_year)
            if g == G3:
                ptr, pcols = pooled_frame(data, cols, pot, mism, train_years)
                va_X = va[cols].copy(); va_X["gid"] = 3
                Q = np.stack([fit_one(ptr[pcols], ptr["y_norm"], va_X[pcols], va[g] / cap,
                                      al, categorical=["gid"]) for al in QUANTS], axis=1) * cap
            else:
                tr = solo_frame(data, cols, g, pot, mism, train_years)
                Q = np.stack([fit_one(tr[cols], tr["_y"], va[cols], va[g], al)
                              for al in QUANTS], axis=1)
            Q = np.clip(np.sort(Q, axis=1), 0, cap)
            # abar: 학습 연도의 평가 대상 평균 발전량 (누수 없음)
            tr_lab = data.dropna(subset=[g])
            tr_lab = tr_lab[tr_lab.index.year.isin(train_years)][g]
            abar = float(tr_lab[tr_lab >= 0.10 * cap].mean())
            preds_med[g] = floor10(Q[:, QUANTS.index(0.60)], cap)   # q60 = v5 기준선
            preds_util[g] = floor10(utility_points(Q, cap, abar), cap)
            actuals[g] = va[g].to_numpy()
            print(f"  [{fold}/{g}] q60기준 {group_score(preds_med[g], actuals[g], cap):.4f} "
                  f"→ 효용최적 {group_score(preds_util[g], actuals[g], cap):.4f}")
        s_med, m_med = score_str(preds_med, actuals, groups)
        s_util, m_util = score_str(preds_util, actuals, groups)
        print(f"  [{fold}] q60 {m_med}")
        print(f"  [{fold}] 효용 {m_util}")
        results[fold] = (s_med, s_util)
    return results


# ──────────────────────── B) 풍속 바이어스 보정 피처 ────────────────────────

def scada_group_ws(labels_index):
    """그룹별 시간 평균 실측 풍속 (SCADA)."""
    scada = _load_scada()
    out = {}
    for g, (maker, turbines) in _GROUP_TURBINES.items():
        d = scada[maker]
        ws = d[[f"{maker}_wtg{t:02d}_ws" for t in turbines]]
        ws = ws.where((ws >= 0) & (ws <= 60))
        hour_end = d.index.ceil("h")
        out[f"{g}_ws"] = ws.mean(axis=1).groupby(hour_end).mean().reindex(labels_index)
    return pd.DataFrame(out, index=labels_index)


def corrector_wshat(data, cols, g, meas, train_years, valid_year):
    """예보 피처 → 실측 풍속 보정 모델. 학습연도는 블록 OOF, 검증연도는 전체학습 예측."""
    params = dict(BASE_PARAMS, objective="l2", seed=RANDOM_SEED)
    y = meas[f"{g}_ws"]
    d_all = data[cols]
    rows_tr = data.index[data.index.year.isin(train_years) & y.reindex(data.index).notna()]
    wshat = pd.Series(np.nan, index=data.index)
    if len(rows_tr) == 0:
        # 학습연도에 SCADA가 없는 그룹(예: 그룹3의 2022) — 피처 NaN 유지
        return wshat
    # 블록 정의: 학습연도 2개 이상이면 연도 교차, 아니면 월 홀짝 교차
    blocks = [rows_tr[rows_tr.year == yr] for yr in train_years]
    blocks = [b for b in blocks if len(b) > 0]
    if len(blocks) < 2:
        blocks = [rows_tr[rows_tr.month % 2 == 0], rows_tr[rows_tr.month % 2 == 1]]
        blocks = [b for b in blocks if len(b) > 0]
    for blk in blocks:
        others = rows_tr.difference(blk)
        if len(others) == 0:
            continue
        m = lgb.train(params, lgb.Dataset(d_all.loc[others], y.reindex(others)), 700)
        wshat.loc[blk] = m.predict(d_all.loc[blk])
    m_full = lgb.train(params, lgb.Dataset(d_all.loc[rows_tr], y.reindex(rows_tr)), 700)
    rows_va = data.index[data.index.year == valid_year]
    wshat.loc[rows_va] = m_full.predict(d_all.loc[rows_va])
    return wshat


def run_B(data, cols, pot, mism):
    print("=== B) NWP→SCADA 풍속 보정 피처 ===")
    meas = scada_group_ws(data.index)
    rated = {"kpx_group_1": 12.0, "kpx_group_2": 12.0, "kpx_group_3": 11.0}
    results = {}
    for fold, (train_years, valid_year, groups) in {
            "fold24": ([2022, 2023], 2024, GROUPS),
            "fold23": ([2022], 2023, G12)}.items():
        preds, actuals = {}, {}
        # 그룹별 보정 풍속 피처 생성 후 데이터 확장
        data2 = data.copy()
        new = []
        for g in GROUPS:
            wshat = corrector_wshat(data, cols, g, meas, train_years, valid_year)
            k = f"wshat_{g[-1]}"
            data2[k] = wshat
            data2[f"{k}_cube"] = wshat ** 3
            ramp = np.clip((wshat - 3.0) / (rated[g] - 3.0), 0, 1) ** 3
            data2[f"{k}_pc"] = ramp
            new += [k, f"{k}_cube", f"{k}_pc"]
        cols2 = cols + new
        for g in groups:
            cap = CAPACITY_KWH[g]
            va = valid_rows(data2, g, valid_year)
            if g == G3:
                ptr, pcols = pooled_frame(data2, cols2, pot, mism, train_years)
                va_X = va[cols2].copy(); va_X["gid"] = 3
                pr = fit_one(ptr[pcols], ptr["y_norm"], va_X[pcols], va[g] / cap, 0.60,
                             categorical=["gid"]) * cap
            else:
                tr = solo_frame(data2, cols2, g, pot, mism, train_years)
                pr = fit_one(tr[cols2], tr["_y"], va[cols2], va[g], 0.60)
            preds[g] = floor10(np.clip(pr, 0, cap), cap)
            actuals[g] = va[g].to_numpy()
            print(f"  [{fold}/{g}] {group_score(preds[g], actuals[g], cap):.4f}")
        s, msg = score_str(preds, actuals, groups)
        print(f"  [{fold}] {msg}")
        results[fold] = s
    return results


if __name__ == "__main__":
    import sys
    data, cols, pot, mism = setup()
    print(f"데이터 {data.shape[0]:,}행, 피처 {len(cols)}개")
    print("v5 기준선(단일시드): fold24 0.6380 / fold23 0.6316\n")
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "A"):
        run_A(data, cols, pot, mism)
        print()
    if only in ("all", "B"):
        run_B(data, cols, pot, mism)
