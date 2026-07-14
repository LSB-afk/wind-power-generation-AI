"""성능 개선 실험 러너.

사용법: python src/exp_runner.py stage1|stage2   (기본 stage2)

- 피처는 cache/ 에 parquet으로 캐시해 반복 실험을 빠르게 한다.
- 채점: fold24(2022~23 학습 → 2024 검증, 3그룹) 기본.
        fold23(2022 학습 → 2023 검증, 그룹1/2)은 후처리 파라미터의 연도 간 안정성 확인용.
- 컨텍스트 피처는 같은 data_available(발표분) 블록 안에서만 계산해 누수를 차단한다.
"""
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, RANDOM_SEED, ROOT, TRAIN_DIR
from features import build_features, _wind_speed
from geo import idw_weights
from metrics import competition_score, group_ficr, group_nmae
from v6_eval import run_stage7

CACHE = ROOT / "cache"

BASE_PARAMS = {
    "objective": "l1",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": RANDOM_SEED,
    "verbosity": -1,
}

LDAPS_RAW_VARS = ["ws10", "ws50max", "blws"]
GFS_RAW_VARS = ["ws10", "ws80", "ws100", "gust"]
CONTEXT_COLS = [
    "ldaps_ws10_mean", "ldaps_ws10_max", "ldaps_ws50max_mean", "ldaps_blws_mean",
    "gfs_ws100_mean", "gfs_ws80_mean", "gfs_ws100_cube_mean",
    "gfs_surface_0_gust_mean", "gfs_ws_pbl_mean",
]
SEEDS3 = (42, 202, 777)


# ──────────────────────────── 캐시 ────────────────────────────

def _slim_raw(path, source):
    df = pd.read_csv(path, encoding="utf-8-sig",
                     parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"])
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"]].copy()
    if source == "ldaps":
        out["ws10"] = _wind_speed(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        out["ws50max"] = _wind_speed(df["heightAboveGround_50_50MUmax"], df["heightAboveGround_50_50MVmax"])
        out["blws"] = _wind_speed(df["heightAboveGround_5_XBLWS"], df["heightAboveGround_5_YBLWS"])
    else:
        out["ws10"] = _wind_speed(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        out["ws80"] = _wind_speed(df["heightAboveGround_80_u"], df["heightAboveGround_80_v"])
        out["ws100"] = _wind_speed(df["heightAboveGround_100_100u"], df["heightAboveGround_100_100v"])
        out["gust"] = df["surface_0_gust"]
    return out


def build_cache():
    CACHE.mkdir(exist_ok=True)
    if not (CACHE / "train_base.parquet").exists():
        print("캐시 생성 중 (최초 1회)...")
        feat = build_features(TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv",
                              context=False)
        feat.to_parquet(CACHE / "train_base.parquet")
        _slim_raw(TRAIN_DIR / "ldaps_train.csv", "ldaps").to_parquet(CACHE / "train_ldaps_raw.parquet")
        _slim_raw(TRAIN_DIR / "gfs_train.csv", "gfs").to_parquet(CACHE / "train_gfs_raw.parquet")
        labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                             parse_dates=["kst_dtm"], index_col="kst_dtm")
        labels.to_parquet(CACHE / "train_labels.parquet")
        print("캐시 완료")


def load_cache():
    feat = pd.read_parquet(CACHE / "train_base.parquet")
    ldaps = pd.read_parquet(CACHE / "train_ldaps_raw.parquet")
    gfs = pd.read_parquet(CACHE / "train_gfs_raw.parquet")
    labels = pd.read_parquet(CACHE / "train_labels.parquet")
    return feat, ldaps, gfs, labels


# ──────────────────────────── 피처 애드온 ────────────────────────────

def add_context(feat: pd.DataFrame, davail: pd.Series) -> pd.DataFrame:
    """같은 발표분(data_available) 24시간 블록 내 lag/lead/rolling — 블록 경계는 NaN."""
    feat = feat.copy()
    block = davail.reindex(feat.index)
    new_cols = {}
    for c in CONTEXT_COLS:
        if c not in feat.columns:
            continue
        g = feat[c].groupby(block)
        for s in (-2, -1, 1, 2):
            new_cols[f"{c}_sh{s}"] = g.shift(s)
        for w in (3, 6):
            roll = g.rolling(w, center=True, min_periods=2)
            new_cols[f"{c}_rm{w}"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{c}_rs{w}"] = roll.std().reset_index(level=0, drop=True)
        new_cols[f"{c}_d1"] = feat[c] - g.shift(1)
    return pd.concat([feat, pd.DataFrame(new_cols, index=feat.index)], axis=1)


def _power_curve(ws, rated_ws):
    """단순 파워커브 근사: cut-in 3 m/s, 정격 도달까지 세제곱 램프."""
    return np.clip((ws - 3.0) / (rated_ws - 3.0), 0.0, 1.0) ** 3


def add_idw(feat: pd.DataFrame, ldaps_raw, gfs_raw) -> pd.DataFrame:
    """터빈 그룹 중심좌표 기준 거리가중 격자 평균 + 파워커브 변환."""
    feat = feat.copy()
    new_cols = {}
    for src, raw, vars_ in (("ldaps", ldaps_raw, LDAPS_RAW_VARS), ("gfs", gfs_raw, GFS_RAW_VARS)):
        grids = raw.drop_duplicates("grid_id")[["grid_id", "latitude", "longitude"]]
        weights = idw_weights(grids)
        for v in vars_:
            wide = raw.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=v)
            for g in GROUPS:
                w = weights[g].reindex(wide.columns).to_numpy()
                col = (wide * w).sum(axis=1).reindex(feat.index)
                new_cols[f"{src}_idw_{g[-1]}_{v}"] = col
    rated = {"kpx_group_1": 12.0, "kpx_group_2": 12.0, "kpx_group_3": 11.0}
    for g in GROUPS:
        ws = new_cols[f"gfs_idw_{g[-1]}_ws100"]
        new_cols[f"pc_{g[-1]}"] = _power_curve(ws, rated[g])
        new_cols[f"gfs_idw_{g[-1]}_ws100_cube"] = ws ** 3
    return pd.concat([feat, pd.DataFrame(new_cols, index=feat.index)], axis=1)


# ──────────────────────────── 학습/평가 ────────────────────────────

def train_predict(tr, va, feature_cols, target, cap, params=None, seeds=(RANDOM_SEED,),
                  weight_fn=None, train_filter_ratio=None):
    p = dict(BASE_PARAMS if params is None else params)
    if train_filter_ratio is not None:
        tr = tr[tr[target] >= train_filter_ratio * cap]
    w = weight_fn(tr[target].to_numpy(), cap) if weight_fn else None
    preds = []
    for s in seeds:
        p["seed"] = s
        dtrain = lgb.Dataset(tr[feature_cols], tr[target], weight=w)
        dvalid = lgb.Dataset(va[feature_cols], va[target], reference=dtrain)
        model = lgb.train(p, dtrain, 5000, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(model.predict(va[feature_cols], num_iteration=model.best_iteration))
    return np.mean(preds, axis=0), model.best_iteration


def run_fold(data, feature_cols, train_years, valid_year, groups=GROUPS,
             bad_mask=None, **kw):
    preds, actuals = {}, {}
    for g in groups:
        d = data.dropna(subset=[g])
        tr = d[d.index.year.isin(train_years)]
        if bad_mask is not None:
            # SCADA 정지/불일치 시간대는 학습에서만 제외 (검증은 전체)
            tr = tr[~bad_mask[f"{g}_bad"].reindex(tr.index).fillna(False)]
        va = d[d.index.year == valid_year]
        pred, _ = train_predict(tr, va, feature_cols, g, CAPACITY_KWH[g], **kw)
        preds[g] = np.clip(pred, 0, CAPACITY_KWH[g])
        actuals[g] = va[g].to_numpy()
    return preds, actuals


def group_score(pred, actual, cap):
    return 0.5 * (1 - group_nmae(pred, actual, cap)) + 0.5 * group_ficr(pred, actual, cap)


def score_str(preds, actuals, groups=GROUPS):
    if set(groups) == set(GROUPS):
        r = competition_score(preds, actuals)
        return r["score"], (f"총점 {r['score']:.4f} (1-NMAE {r['one_minus_nmae']:.4f}, "
                            f"FICR {r['ficr']:.4f})")
    nm = np.mean([group_nmae(preds[g], actuals[g], CAPACITY_KWH[g]) for g in groups])
    fi = np.mean([group_ficr(preds[g], actuals[g], CAPACITY_KWH[g]) for g in groups])
    s = 0.5 * (1 - nm) + 0.5 * fi
    return s, f"부분총점 {s:.4f} (1-NMAE {1-nm:.4f}, FICR {fi:.4f})"


# ──────────────────────────── 후처리 튜닝 ────────────────────────────

def tune_post(pred, actual, cap):
    """그룹별 (scale, floor) 그리드서치 — 그룹 총점 기여 최대화."""
    best = (-1.0, 1.0, 0.0)
    for sc in np.arange(0.90, 1.401, 0.025):
        for fl in np.arange(0.0, 0.201, 0.005) * cap:
            p = np.clip(np.maximum(pred * sc, fl), 0, cap)
            s = group_score(p, actual, cap)
            if s > best[0]:
                best = (s, sc, fl)
    return best  # (score, scale, floor)


def apply_post(pred, cap, scale, floor):
    return np.clip(np.maximum(pred * scale, floor), 0, cap)


# ──────────────────────────── 실험 ────────────────────────────

def stage1(data_by_name):
    results = {}
    for name, (data, cols) in data_by_name.items():
        preds, actuals = run_fold(data, cols, [2022, 2023], 2024)
        s, msg = score_str(preds, actuals)
        results[name] = s
        print(f"[{name}] {msg}  (피처 {len(cols)}개)")
    best = max(results, key=results.get)
    print(f"\n최고 피처셋: {best} = {results[best]:.4f}")


def stage2(data, cols):
    log = []

    def report(name, preds, actuals, groups=GROUPS):
        s, msg = score_str(preds, actuals, groups)
        log.append((name, s))
        print(f"[{name}] {msg}")
        return s

    # S0: ctx 피처 기준선 (단일 시드)
    p0, a0 = run_fold(data, cols, [2022, 2023], 2024)
    report("S0_ctx기준선", p0, a0)

    # S1: 3시드 앙상블
    p1, _ = run_fold(data, cols, [2022, 2023], 2024, seeds=SEEDS3)
    report("S1_3시드", p1, a0)

    # S2: FICR 지향 샘플 가중치 (발전량 큰 시간대 강조)
    p2, _ = run_fold(data, cols, [2022, 2023], 2024,
                     weight_fn=lambda y, cap: 1.0 + 2.0 * y / cap)
    report("S2_샘플가중치", p2, a0)

    # S3: 저발전 구간 학습 제외 (평가 필터와 정렬)
    p3, _ = run_fold(data, cols, [2022, 2023], 2024, train_filter_ratio=0.05)
    report("S3_저발전제외(5%)", p3, a0)

    # S4/S5: 하이퍼파라미터
    hp1 = dict(BASE_PARAMS, num_leaves=127, min_data_in_leaf=20,
               learning_rate=0.02, feature_fraction=0.6)
    p4, _ = run_fold(data, cols, [2022, 2023], 2024, params=hp1)
    report("S4_hp깊게", p4, a0)
    hp2 = dict(BASE_PARAMS, num_leaves=255, min_data_in_leaf=10,
               learning_rate=0.015, feature_fraction=0.5, lambda_l2=3.0)
    p5, _ = run_fold(data, cols, [2022, 2023], 2024, params=hp2)
    report("S5_hp더깊게", p5, a0)

    # S6: 그룹 통합(pooled) 학습 — 그룹3 데이터 부족 보완
    pooled_frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g])
        f = d[cols].copy()
        f["y_norm"] = d[g] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        pooled_frames.append(f)
    pooled = pd.concat(pooled_frames)
    pcols = cols + ["gid"]
    ptr = pooled[pooled.index.year <= 2023]
    pva3 = pooled[(pooled.index.year == 2024) & (pooled["gid"] == 3)]
    dtrain = lgb.Dataset(ptr[pcols], ptr["y_norm"],
                         categorical_feature=["gid"])
    dvalid = lgb.Dataset(pva3[pcols], pva3["y_norm"], reference=dtrain)
    m = lgb.train(BASE_PARAMS, dtrain, 5000, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    g3_pred = np.clip(m.predict(pva3[pcols]) * CAPACITY_KWH["kpx_group_3"],
                      0, CAPACITY_KWH["kpx_group_3"])
    g3_actual = a0["kpx_group_3"]
    s_pool = group_score(g3_pred, g3_actual, CAPACITY_KWH["kpx_group_3"])
    s_solo = group_score(p0["kpx_group_3"], g3_actual, CAPACITY_KWH["kpx_group_3"])
    print(f"[S6_그룹통합] 그룹3 단독점수: solo {s_solo:.4f} → pooled {s_pool:.4f}")
    log.append(("S6_g3pooled(그룹3만)", s_pool))

    # S7: 후처리 (scale, floor) — fold24 직접 튜닝(참고치) + fold23→24 이전 안정성
    print("\n--- S7 후처리 튜닝 ---")
    tuned = {}
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        s_tuned, sc, fl = tune_post(p1[g], a0[g], cap)
        s_raw = group_score(p1[g], a0[g], cap)
        tuned[g] = (sc, fl)
        print(f"  {g}: {s_raw:.4f} → {s_tuned:.4f}  (scale={sc:.3f}, floor={fl/cap:.3f}cap)")
    p_post = {g: apply_post(p1[g], CAPACITY_KWH[g], *tuned[g]) for g in GROUPS}
    report("S7_3시드+후처리(24튜닝)", p_post, a0)

    # 연도 간 이전 안정성: 2023에서 튜닝 → 2024에 적용 (그룹1/2)
    g12 = ["kpx_group_1", "kpx_group_2"]
    p23, a23 = run_fold(data, cols, [2022], 2023, groups=g12, seeds=SEEDS3)
    for g in g12:
        cap = CAPACITY_KWH[g]
        _, sc23, fl23 = tune_post(p23[g], a23[g], cap)
        s_raw24 = group_score(p1[g], a0[g], cap)
        s_tr24 = group_score(apply_post(p1[g], cap, sc23, fl23), a0[g], cap)
        print(f"  [이전성] {g}: 23년튜닝(scale={sc23:.3f}, floor={fl23/cap:.3f}cap) "
              f"→ 24년 적용 {s_raw24:.4f} → {s_tr24:.4f}")

    print("\n===== stage2 요약 =====")
    for name, s in sorted(log, key=lambda x: -x[1]):
        print(f"  {name}: {s:.4f}")


def stage3(data, cols):
    """조합 실험: 학습 레시피 x 후처리. 후처리는 fold24 튜닝(최종 채택용)과
    fold23→24 이전(연도 간 일반화 확인) 둘 다 본다."""
    a0 = {g: data.dropna(subset=[g]).pipe(
        lambda d: d[d.index.year == 2024][g].to_numpy()) for g in GROUPS}

    recipes = {
        "C1_filter05": dict(train_filter_ratio=0.05),
        "C2_weight": dict(weight_fn=lambda y, cap: 1.0 + 2.0 * y / cap),
        "C3_filter05+weight": dict(train_filter_ratio=0.05,
                                   weight_fn=lambda y, cap: 1.0 + 2.0 * y / cap),
        "C4_filter08": dict(train_filter_ratio=0.08),
        "C5_q55+filter05": dict(train_filter_ratio=0.05,
                                params=dict(BASE_PARAMS, objective="quantile", alpha=0.55)),
        "C6_q60+filter05": dict(train_filter_ratio=0.05,
                                params=dict(BASE_PARAMS, objective="quantile", alpha=0.60)),
    }

    results = {}
    for name, kw in recipes.items():
        preds, _ = run_fold(data, cols, [2022, 2023], 2024, **kw)
        s_raw, msg = score_str(preds, a0)
        # fold24 직접 튜닝 후처리
        tuned = {g: tune_post(preds[g], a0[g], CAPACITY_KWH[g]) for g in GROUPS}
        p_post = {g: apply_post(preds[g], CAPACITY_KWH[g], tuned[g][1], tuned[g][2])
                  for g in GROUPS}
        s_post, _ = score_str(p_post, a0)
        results[name] = (s_raw, s_post, tuned, preds)
        pp = ", ".join(f"g{g[-1]}(sc={tuned[g][1]:.2f},fl={tuned[g][2]/CAPACITY_KWH[g]:.3f})"
                       for g in GROUPS)
        print(f"[{name}] raw {s_raw:.4f} → 후처리 {s_post:.4f}  [{pp}]")

    best = max(results, key=lambda k: results[k][1])
    print(f"\n최고 레시피: {best} (후처리 포함 {results[best][1]:.4f})")

    # 최고 레시피의 연도 간 이전성 확인 (그룹1/2: 2023 튜닝 → 2024 적용)
    g12 = ["kpx_group_1", "kpx_group_2"]
    kw = recipes[best]
    p23, a23 = run_fold(data, cols, [2022], 2023, groups=g12, **kw)
    p24 = results[best][3]
    for g in g12:
        cap = CAPACITY_KWH[g]
        _, sc23, fl23 = tune_post(p23[g], a23[g], cap)
        s_raw24 = group_score(p24[g], a0[g], cap)
        s_tr = group_score(apply_post(p24[g], cap, sc23, fl23), a0[g], cap)
        s_oracle = results[best][2][g][0]
        print(f"  [이전성] {g}: 23년튜닝(sc={sc23:.2f},fl={fl23/cap:.3f}) → 24년 "
              f"{s_raw24:.4f} → {s_tr:.4f} (24년직접튜닝 {s_oracle:.4f})")

    # 그룹3: 통합학습 + 최고 레시피 조합
    print("\n--- 그룹3 통합학습 x 최고 레시피 ---")
    pooled_frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g])
        f = d[cols].copy()
        f["y_norm"] = d[g] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        pooled_frames.append(f)
    pooled = pd.concat(pooled_frames)
    pcols = cols + ["gid"]
    kw_best = dict(recipes[best])
    params = kw_best.pop("params", BASE_PARAMS)
    fr = kw_best.pop("train_filter_ratio", None)
    wfn = kw_best.pop("weight_fn", None)
    ptr = pooled[pooled.index.year <= 2023]
    if fr is not None:
        ptr = ptr[ptr["y_norm"] >= fr]
    w = wfn(ptr["y_norm"].to_numpy(), 1.0) if wfn else None
    pva3 = pooled[(pooled.index.year == 2024) & (pooled["gid"] == 3)]
    dtrain = lgb.Dataset(ptr[pcols], ptr["y_norm"], weight=w, categorical_feature=["gid"])
    dvalid = lgb.Dataset(pva3[pcols], pva3["y_norm"], reference=dtrain)
    m = lgb.train(params, dtrain, 5000, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    cap3 = CAPACITY_KWH["kpx_group_3"]
    g3_pred = np.clip(m.predict(pva3[pcols]) * cap3, 0, cap3)
    s3_raw = group_score(g3_pred, a0["kpx_group_3"], cap3)
    s3t, sc3, fl3 = tune_post(g3_pred, a0["kpx_group_3"], cap3)
    s3_solo_post = results[best][2]["kpx_group_3"][0]
    print(f"  그룹3: solo+후처리 {s3_solo_post:.4f} vs pooled raw {s3_raw:.4f} "
          f"→ pooled+후처리 {s3t:.4f} (sc={sc3:.2f},fl={fl3/cap3:.3f})")

    # 최종 조합 총점 (그룹1/2 solo최고 + 그룹3 pooled, 모두 후처리)
    final_preds = {g: apply_post(p24[g], CAPACITY_KWH[g], results[best][2][g][1],
                                 results[best][2][g][2]) for g in g12}
    final_preds["kpx_group_3"] = apply_post(g3_pred, cap3, sc3, fl3)
    s, msg = score_str(final_preds, a0)
    print(f"\n[최종조합] {msg}")


def stage5(datasets):
    """v4: 이중 폴드(22→23, 22-23→24) 견고성 실험.

    v2/v3의 Public 전이 실패 교훈: 단일 폴드(2024)에서만 이긴 것은 채택하지 않는다.
    안전 하한 floor=0.10cap은 평가 필터(실발전 >= 10%cap) 정의상 어떤 해에도
    채점 시간대를 악화시킬 수 없는 무손실 후처리이므로 항상 적용 가능.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    data_base, cols_base = datasets["base"]
    data, cols = datasets["ctx"]
    bad = pd.read_parquet(CACHE / "scada_badmask.parquet")
    g12 = ["kpx_group_1", "kpx_group_2"]

    def floor10(preds):
        return {g: apply_post(p, CAPACITY_KWH[g], 1.0, 0.10 * CAPACITY_KWH[g])
                for g, p in preds.items()}

    def dual_eval(name, kw, use_base=False, post=False):
        d, c = (data_base, cols_base) if use_base else (data, cols)
        p24, a24 = run_fold(d, c, [2022, 2023], 2024, **kw)
        p23, a23 = run_fold(d, c, [2022], 2023, groups=g12, **kw)
        if post:
            p24, p23 = floor10(p24), floor10(p23)
        s24, m24 = score_str(p24, a24)
        s23, m23 = score_str(p23, a23, g12)
        per_g = " ".join(f"g{g[-1]}={group_score(p24[g], a24[g], CAPACITY_KWH[g]):.4f}"
                         for g in GROUPS)
        print(f"[{name}] fold24 {m24} | fold23 {m23}\n         fold24 그룹별: {per_g}")
        return p24, a24, s24, s23

    print("=== stage5: 이중 폴드 견고성 실험 ===")
    dual_eval("R0_v1레시피(L1+base)", {}, use_base=True)
    dual_eval("R1_L1+ctx", {})
    dual_eval("R2_R1+SCADA클리닝", dict(bad_mask=bad))
    p24_r3, a24, _, _ = dual_eval("R3_R2+floor10", dict(bad_mask=bad), post=True)
    dual_eval("R4_q60+filter05+클리닝+floor10",
              dict(bad_mask=bad, train_filter_ratio=0.05,
                   params=dict(BASE_PARAMS, objective="quantile", alpha=0.60)),
              post=True)

    # R5: HistGB(absolute_error) 블렌드 — 모델 다양성 (R3 위에)
    print("--- R5: LGBM+HistGB 블렌드 (fold24/23) ---")
    for years_tr, year_va, groups in ([[2022, 2023], 2024, GROUPS], [[2022], 2023, g12]):
        preds_l, actuals = run_fold(data, cols, years_tr, year_va, groups=groups, bad_mask=bad)
        preds_b = {}
        for g in groups:
            d = data.dropna(subset=[g])
            tr = d[d.index.year.isin(years_tr)]
            tr = tr[~bad[f"{g}_bad"].reindex(tr.index).fillna(False)]
            va = d[d.index.year == year_va]
            hgb = HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=2000, learning_rate=0.05,
                max_leaf_nodes=63, early_stopping=True, n_iter_no_change=50,
                random_state=RANDOM_SEED)
            hgb.fit(tr[cols], tr[g])
            pb = np.clip(hgb.predict(va[cols]), 0, CAPACITY_KWH[g])
            preds_b[g] = np.clip(0.5 * preds_l[g] + 0.5 * pb, 0, CAPACITY_KWH[g])
        s, msg = score_str(floor10(preds_b), actuals, groups)
        print(f"  fold{str(year_va)[-2:]}: {msg}")


def stage6(datasets):
    """v5 후보 실험 — 이중 폴드(22→23, 22-23→24) 일관 승리만 채택.

    T1: potential 라벨(가용률 보정) — 정지 시간대를 버리지 않고 복원
    T2: 그룹1/2도 통합(pooled) 학습
    T3: 분위 앙상블 (alpha 0.55/0.60/0.65 평균)
    T4: 스프레드/불확실성 피처 팩
    """
    data, cols = datasets["ctx"]
    bad = pd.read_parquet(CACHE / "scada_badmask.parquet")
    mism = pd.read_parquet(CACHE / "scada_mismatch.parquet")
    pot = pd.read_parquet(CACHE / "scada_potential.parquet")
    g12 = ["kpx_group_1", "kpx_group_2"]
    Q60 = dict(BASE_PARAMS, objective="quantile", alpha=0.60)

    def floor10(preds):
        return {g: apply_post(p, CAPACITY_KWH[g], 1.0, 0.10 * CAPACITY_KWH[g])
                for g, p in preds.items()}

    def report(name, p24, a24, p23, a23):
        s24, m24 = score_str(floor10(p24), a24)
        s23, m23 = score_str(floor10(p23), a23, g12)
        per_g = " ".join(f"g{g[-1]}={group_score(floor10(p24)[g], a24[g], CAPACITY_KWH[g]):.4f}"
                         for g in GROUPS)
        print(f"[{name}] fold24 {m24} | fold23 {m23}\n         fold24 그룹별: {per_g}")

    def v4_fold(train_years, valid_year, groups, params=None, seeds=(RANDOM_SEED,)):
        return run_fold(data, cols, train_years, valid_year, groups=groups, bad_mask=bad,
                        train_filter_ratio=0.05, params=params or Q60, seeds=seeds)

    # ── T0: v4 기준선 재계측 ──
    p24, a24 = v4_fold([2022, 2023], 2024, GROUPS)
    p23, a23 = v4_fold([2022], 2023, g12)
    report("T0_v4기준선", p24, a24, p23, a23)

    # ── T1: potential 라벨 (a: 가용률 계수 적용 / b: 미적용) ──
    def potential_fold(train_years, valid_year, groups, use_avail):
        preds, actuals = {}, {}
        for g in groups:
            cap = CAPACITY_KWH[g]
            y_pot = pot[f"{g}_potential"]
            d = data.dropna(subset=[g])
            tr = d[d.index.year.isin(train_years)].copy()
            tr["_y"] = y_pot.reindex(tr.index)
            tr = tr[~mism[f"{g}_mismatch"].reindex(tr.index).fillna(False)]
            tr = tr.dropna(subset=["_y"])
            tr = tr[tr["_y"] >= 0.05 * cap]
            va = d[d.index.year == valid_year]
            avail = 1.0
            if use_avail:
                m = (~mism[f"{g}_mismatch"].reindex(tr.index).fillna(False))
                avail = float(tr.loc[m, g].sum() / tr.loc[m, "_y"].sum())
            dtrain = lgb.Dataset(tr[cols], tr["_y"])
            m2 = lgb.train(dict(Q60, seed=RANDOM_SEED), dtrain, 5000,
                           valid_sets=[lgb.Dataset(va[cols], va[g], reference=dtrain)],
                           callbacks=[lgb.early_stopping(200, verbose=False)])
            pred = m2.predict(va[cols], num_iteration=m2.best_iteration) * avail
            preds[g] = np.clip(pred, 0, cap)
            actuals[g] = va[g].to_numpy()
        return preds, actuals

    for label, use_avail in (("T1a_potential+가용률계수", True), ("T1b_potential(계수없음)", False)):
        q24, b24 = potential_fold([2022, 2023], 2024, GROUPS, use_avail)
        q23, b23 = potential_fold([2022], 2023, g12, use_avail)
        report(label, q24, b24, q23, b23)

    # ── T2: 그룹1/2도 pooled ──
    def pooled_fold(train_years, valid_year, groups):
        frames = []
        for g in GROUPS:
            d = data.dropna(subset=[g])
            d = d[~bad[f"{g}_bad"].reindex(d.index).fillna(False)]
            f = d[cols].copy()
            f["y_norm"] = d[g] / CAPACITY_KWH[g]
            f["gid"] = int(g[-1])
            frames.append(f)
        pooled = pd.concat(frames)
        pcols = cols + ["gid"]
        ptr = pooled[pooled.index.year.isin(train_years)]
        ptr = ptr[ptr["y_norm"] >= 0.05]
        preds, actuals = {}, {}
        for g in groups:
            cap = CAPACITY_KWH[g]
            d = data.dropna(subset=[g])
            va = d[d.index.year == valid_year]
            va_X = va[cols].copy(); va_X["gid"] = int(g[-1])
            dtrain = lgb.Dataset(ptr[pcols], ptr["y_norm"], categorical_feature=["gid"])
            m = lgb.train(dict(Q60, seed=RANDOM_SEED), dtrain, 5000,
                          valid_sets=[lgb.Dataset(va_X[pcols], va[g] / cap, reference=dtrain)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            preds[g] = np.clip(m.predict(va_X[pcols], num_iteration=m.best_iteration) * cap, 0, cap)
            actuals[g] = va[g].to_numpy()
        return preds, actuals

    r24, c24 = pooled_fold([2022, 2023], 2024, GROUPS)
    r23, c23 = pooled_fold([2022], 2023, g12)
    report("T2_전그룹pooled", r24, c24, r23, c23)

    # ── T3: 분위 앙상블 (0.55/0.60/0.65) ──
    def alpha_fold(train_years, valid_year, groups):
        acc = None
        for al in (0.55, 0.60, 0.65):
            p, a = v4_fold(train_years, valid_year, groups,
                           params=dict(BASE_PARAMS, objective="quantile", alpha=al))
            acc = p if acc is None else {g: acc[g] + p[g] for g in groups}
        return {g: acc[g] / 3 for g in groups}, a

    s24, d24 = alpha_fold([2022, 2023], 2024, GROUPS)
    s23, d23 = alpha_fold([2022], 2023, g12)
    report("T3_분위앙상블", s24, d24, s23, d23)

    # ── T4: 스프레드/불확실성 피처 팩 ──
    feat2 = data[cols].copy()
    eps = 0.1
    feat2["x_spread_ws"] = feat2["gfs_ws100_mean"] - feat2["ldaps_ws50max_mean"]
    feat2["x_spread_ws10"] = feat2["gfs_ws10_mean"] - feat2["ldaps_ws10_mean"]
    feat2["x_cv_gfs"] = feat2["gfs_ws100_std"] / (feat2["gfs_ws100_mean"] + eps)
    feat2["x_cv_ldaps"] = feat2["ldaps_ws10_std"] / (feat2["ldaps_ws10_mean"] + eps)
    sin_cols = [c for c in cols if c.startswith("gfs_g") and c.endswith("d100_wd_sin")]
    cos_cols = [c for c in cols if c.startswith("gfs_g") and c.endswith("d100_wd_cos")]
    feat2["x_wd_sin"] = feat2[sin_cols].mean(axis=1)
    feat2["x_wd_cos"] = feat2[cos_cols].mean(axis=1)
    feat2["x_ws_x_sin"] = feat2["gfs_ws100_mean"] * feat2["x_wd_sin"]
    feat2["x_ws_x_cos"] = feat2["gfs_ws100_mean"] * feat2["x_wd_cos"]
    data2 = feat2.join(data[GROUPS])
    cols2 = list(feat2.columns)
    t24, e24 = run_fold(data2, cols2, [2022, 2023], 2024, bad_mask=bad,
                        train_filter_ratio=0.05, params=Q60)
    t23, e23 = run_fold(data2, cols2, [2022], 2023, groups=g12, bad_mask=bad,
                        train_filter_ratio=0.05, params=Q60)
    report("T4_피처팩", t24, e24, t23, e23)


FINAL_RECIPE = dict(train_filter_ratio=0.05,
                    params=dict(BASE_PARAMS, objective="quantile", alpha=0.60))


def _pooled_g3_predict(data, cols, train_years, valid_year, seeds=(RANDOM_SEED,)):
    """그룹 통합 학습 후 그룹3 예측 (FINAL_RECIPE 적용)."""
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g])
        f = d[cols].copy()
        f["y_norm"] = d[g] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        frames.append(f)
    pooled = pd.concat(frames)
    pcols = cols + ["gid"]
    ptr = pooled[pooled.index.year.isin(train_years)]
    ptr = ptr[ptr["y_norm"] >= FINAL_RECIPE["train_filter_ratio"]]
    pva = pooled[(pooled.index.year == valid_year) & (pooled["gid"] == 3)]
    preds = []
    for s in seeds:
        p = dict(FINAL_RECIPE["params"], seed=s)
        dtrain = lgb.Dataset(ptr[pcols], ptr["y_norm"], categorical_feature=["gid"])
        dvalid = lgb.Dataset(pva[pcols], pva["y_norm"], reference=dtrain)
        m = lgb.train(p, dtrain, 5000, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(pva[pcols], num_iteration=m.best_iteration))
    cap3 = CAPACITY_KWH["kpx_group_3"]
    return np.clip(np.mean(preds, axis=0) * cap3, 0, cap3), pva.index


def stage4(data, cols):
    """최종 레시피(3시드) 안정성 + 후처리 파라미터 확정."""
    g12 = ["kpx_group_1", "kpx_group_2"]
    a24 = {g: data.dropna(subset=[g]).pipe(
        lambda d: d[d.index.year == 2024][g].to_numpy()) for g in GROUPS}

    # 그룹1/2 solo (3시드) + 그룹3 pooled (3시드)
    p24, _ = run_fold(data, cols, [2022, 2023], 2024, groups=g12, seeds=SEEDS3, **FINAL_RECIPE)
    g3_pred, _ = _pooled_g3_predict(data, cols, [2022, 2023], 2024, seeds=SEEDS3)
    p24["kpx_group_3"] = g3_pred
    s_raw, msg = score_str(p24, a24)
    print(f"[최종레시피 3시드 raw] {msg}")

    # 후처리 파라미터: Public LB에서 FICR 전이가 약했으므로 그룹1/2는
    # 2024 직접 튜닝값이 23·24 평균보다 덜 공격적이고 점수가 낮지 않으면 직접값을 채택한다.
    p23, a23 = run_fold(data, cols, [2022], 2023, groups=g12, seeds=SEEDS3, **FINAL_RECIPE)
    post = {}
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        _, sc24, fl24 = tune_post(p24[g], a24[g], cap)
        if g in g12:
            _, sc23, fl23 = tune_post(p23[g], a23[g], cap)
            sc_avg, fl_avg = (sc24 + sc23) / 2, (fl24 + fl23) / 2
            s_with24 = group_score(apply_post(p24[g], cap, sc24, fl24), a24[g], cap)
            s_withavg = group_score(apply_post(p24[g], cap, sc_avg, fl_avg), a24[g], cap)
            if sc24 <= sc_avg and fl24 <= fl_avg and s_with24 >= s_withavg:
                chosen = (sc24, fl24)
                reason = "24직접"
            else:
                chosen = (sc_avg, fl_avg)
                reason = "23·24평균"
            print(f"  {g}: 24튜닝(sc={sc24:.2f},fl={fl24/cap:.3f})={s_with24:.4f}  "
                  f"평균(sc={sc_avg:.2f},fl={fl_avg/cap:.3f})={s_withavg:.4f}  "
                  f"→ 채택={reason}")
            post[g] = chosen
        else:
            # 그룹3: 2024 상/하반기 교차 안정성 확인
            n = len(g3_pred)
            h1, h2 = slice(0, n // 2), slice(n // 2, n)
            cap3 = cap
            _, sch1, flh1 = tune_post(g3_pred[h1], a24[g][h1], cap3)
            s_h2_tr = group_score(apply_post(g3_pred[h2], cap3, sch1, flh1), a24[g][h2], cap3)
            s_h2_or, _, _ = tune_post(g3_pred[h2], a24[g][h2], cap3)
            print(f"  {g}: H1튜닝(sc={sch1:.2f},fl={flh1/cap3:.3f}) → H2적용 {s_h2_tr:.4f} "
                  f"(H2직접 {s_h2_or:.4f}) | 전체24튜닝(sc={sc24:.2f},fl={fl24/cap:.3f})")
            post[g] = (sc24, fl24)

    p_final = {g: apply_post(p24[g], CAPACITY_KWH[g], *post[g]) for g in GROUPS}
    s, msg = score_str(p_final, a24)
    print(f"\n[최종(채택 후처리)] {msg}")
    print("채택 후처리 파라미터:", {g: (round(post[g][0], 3), round(post[g][1], 1))
                                    for g in GROUPS})


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stage3"
    if mode == "stage7":
        seeds = SEEDS3
        baseline_only = False
        screen = None
        args = sys.argv[2:]
        position = 0
        while position < len(args):
            argument = args[position]
            if argument == "--baseline-only":
                baseline_only = True
                position += 1
                continue
            if argument == "--seeds":
                position += 1
                seed_values = []
                while position < len(args) and not args[position].startswith("--"):
                    seed_values.append(args[position])
                    position += 1
                if not seed_values:
                    print("--seeds requires at least one integer", file=sys.stderr)
                    return 2
                try:
                    seeds = tuple(int(value) for value in seed_values)
                except ValueError:
                    print(
                        f"invalid --seeds value: {' '.join(seed_values)}",
                        file=sys.stderr,
                    )
                    return 2
                continue
            if argument == "--screen":
                position += 1
                if position >= len(args) or args[position].startswith("--"):
                    print("--screen requires one value", file=sys.stderr)
                    return 2
                screen = args[position]
                position += 1
                continue
            print(f"unknown stage7 argument: {argument}", file=sys.stderr)
            return 2
        if screen is None:
            return run_stage7(seeds, baseline_only=baseline_only)
        return run_stage7(seeds, baseline_only=baseline_only, screen=screen)
    if mode not in {"stage1", "stage2", "stage3", "stage4", "stage5", "stage6"}:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2

    build_cache()
    base_feat, ldaps_raw, gfs_raw, labels = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm"
    )["data_available_kst_dtm"]

    feat_ctx = add_context(base_feat, davail)

    def dataset(feat):
        return feat.join(labels, how="inner"), list(feat.columns)

    if mode == "stage1":
        feat_full = add_idw(feat_ctx, ldaps_raw, gfs_raw)
        stage1({"E0_base": dataset(base_feat), "E1_ctx": dataset(feat_ctx),
                "E2_ctx_idw": dataset(feat_full)})
    elif mode == "stage2":
        data, cols = dataset(feat_ctx)
        stage2(data, cols)
    elif mode == "stage3":
        data, cols = dataset(feat_ctx)
        stage3(data, cols)
    elif mode == "stage4":
        data, cols = dataset(feat_ctx)
        stage4(data, cols)
    elif mode == "stage5":
        stage5({"base": dataset(base_feat), "ctx": dataset(feat_ctx)})
    elif mode == "stage6":
        stage6({"ctx": dataset(feat_ctx)})


if __name__ == "__main__":
    raise SystemExit(main())
