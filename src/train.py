"""학습 스크립트 (대회 규정: 추론 코드와 분리).

v9 레시피 — 서로 다른 3개 피처 표현의 앙상블.

배경: v4~v8 은 단일 표현을 계속 손봤고, 로컬 홀드아웃은 올랐지만 Public 은
0.635 에서 정체했다. 블록 부트스트랩으로 재보니 단일 폴드 점수의 표준오차가
±0.0063 이라 그동안의 절대점수 비교 자체가 무의미했다. v9 는 (a) 문헌 기반으로
전처리를 재설계하고, (b) 하나를 고르는 대신 세 표현을 평균해 분포 이동에 대한
분산을 줄인다.

세 표현:
  base     : v8 피처 (348) — 격자별 바람 + 집계 + 물리 팩
  full     : 전 변수 x 전 격자 (1577) — 허브 외삽 기점을 인접층(GFS 80/100 m,
             LDAPS 50 m)으로 교정, 가온도 공기밀도, 벌크 리처드슨 수, 선행시간
  fullterr : full + 풍향 섹터별 지형 보정 풍속·SCADA 경험적 파워커브 (1590)

공통 레시피(v5 계승, 재검증 완료):
  potential 학습 타깃 / quantile(0.60) / 저발전 5%cap 학습 제외 /
  그룹1·2 단독 + 그룹3 통합(pooled) / 시드 3개 / floor = 0.10 x 설비용량

검증 근거: 이중 폴드 페어드 블록 부트스트랩(7일 블록 x 2000)
  fold24 +0.0035 (P(개선)=99.0%, 평균>2SE 통과) / fold23 +0.0012 (P=85.9%)
  월 블록 부호검정 18/24 승 (p≈0.011)
"""
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, ROOT, TRAIN_DIR
from exp_runner import BASE_PARAMS, add_context, load_cache
from features import build_features
from metrics import competition_score
from postprocess import apply_post
from scada import build_mismatch_mask, build_potential
from terrain import apply_terrain, fit_terrain, save_terrain

CACHE = ROOT / "cache"
REPS = ("base", "full", "fullterr")
SEEDS = (42, 202, 777)
ALPHA, FILTER, FLOOR = 0.60, 0.05, 0.10
ALL_YEARS = [2022, 2023, 2024]
HOLDOUT_TRAIN, HOLDOUT_VALID = [2022, 2023], 2024
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


def params(n_feat, seed):
    ff = 0.7 if n_feat < 500 else (0.35 if n_feat < 1000 else 0.25)
    return dict(BASE_PARAMS, objective="quantile", alpha=ALPHA,
                feature_fraction=ff, seed=seed)


def build_train_reps(fit_years):
    """학습용 3개 표현. 지형 보정표는 fit_years 로만 적합한다."""
    base, ldaps_raw, _, _ = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    reps = {"base": add_context(base, davail).join(
        pd.read_parquet(CACHE / "train_phys.parquet"), how="left")}
    reps["full"] = pd.read_parquet(CACHE / "full_train.parquet")
    tparams = fit_terrain(reps["full"], fit_years)
    reps["fullterr"] = reps["full"].join(apply_terrain(reps["full"], tparams), how="left")
    return reps, tparams


def solo_frame(data, g, pot, mism, years):
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


def fit_seeds(tr_X, tr_y, va=None, rounds=None, cat=None):
    """va 주면 조기종료로 (예측, 라운드수), rounds 주면 고정 라운드 학습(모델 리스트)."""
    preds, iters, models = [], [], []
    for s in SEEDS:
        p = params(tr_X.shape[1], s)
        dtr = lgb.Dataset(tr_X, tr_y, categorical_feature=list(cat) if cat else "auto")
        if va is not None:
            m = lgb.train(p, dtr, 5000,
                          valid_sets=[lgb.Dataset(va[0], va[1], reference=dtr)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            preds.append(m.predict(va[0], num_iteration=m.best_iteration))
            iters.append(m.best_iteration)
        else:
            models.append(lgb.train(p, dtr, rounds))
    if va is not None:
        return np.mean(preds, axis=0), int(np.mean(iters))
    return models


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"], index_col="kst_dtm")
    pot, mism = build_potential(labels), build_mismatch_mask(labels)

    # ── 1) 2024 홀드아웃: 검증 점수 + 최종 학습에 쓸 라운드 수 ──
    print("[1/2] 2024 홀드아웃 검증")
    reps_ho, _ = build_train_reps(HOLDOUT_TRAIN)
    preds, actual, rounds = {g: [] for g in GROUPS}, {}, {}
    for rep in REPS:
        feat = reps_ho[rep]
        data = feat.join(labels, how="inner")
        cols = list(feat.columns)
        for g in G12:
            cap = CAPACITY_KWH[g]
            va = data.dropna(subset=[g])
            va = va[va.index.year == HOLDOUT_VALID]
            tr = solo_frame(data, g, pot, mism, HOLDOUT_TRAIN)
            pr, it = fit_seeds(tr[cols], tr["_y"], va=(va[cols], va[g]))
            preds[g].append(np.clip(pr, 0, cap))
            actual[g] = va[g].to_numpy()
            rounds[f"{rep}|{g}"] = it
        ptr, pcols = pooled_frame(data, cols, pot, mism, HOLDOUT_TRAIN)
        cap3 = CAPACITY_KWH[G3]
        va3 = data.dropna(subset=[G3])
        va3 = va3[va3.index.year == HOLDOUT_VALID]
        vx = va3[cols].copy()
        vx["gid"] = 3
        pr, it = fit_seeds(ptr[pcols], ptr["y_norm"], va=(vx[pcols], va3[G3] / cap3),
                           cat=["gid"])
        preds[G3].append(np.clip(pr * cap3, 0, cap3))
        actual[G3] = va3[G3].to_numpy()
        rounds[f"{rep}|pooled"] = it
        print(f"  {rep}: 완료", flush=True)

    final = {g: apply_post(np.mean(preds[g], axis=0), CAPACITY_KWH[g],
                           1.0, FLOOR * CAPACITY_KWH[g]) for g in GROUPS}
    r = competition_score(final, actual)
    print("\n===== 2024 홀드아웃 (3표현 앙상블 + floor10) =====")
    for g in GROUPS:
        print(f"  {g}: NMAE={r['detail'][g]['nmae']:.4f}  FICR={r['detail'][g]['ficr']:.4f}")
    print(f"  1-NMAE = {r['one_minus_nmae']:.4f}\n  FICR   = {r['ficr']:.4f}"
          f"\n  총점   = {r['score']:.4f}")

    # ── 2) 전체 데이터(2022~2024) 재학습 및 저장 ──
    print("\n[2/2] 전체 데이터 재학습")
    reps_all, tparams = build_train_reps(ALL_YEARS)
    save_terrain(tparams, MODEL_DIR / "terrain_params.npz")
    feature_cols = {}
    for rep in REPS:
        feat = reps_all[rep]
        data = feat.join(labels, how="inner")
        cols = list(feat.columns)
        feature_cols[rep] = cols
        for g in G12:
            tr = solo_frame(data, g, pot, mism, ALL_YEARS)
            n = max(int(rounds[f"{rep}|{g}"] * 1.2), 100)
            for s, m in zip(SEEDS, fit_seeds(tr[cols], tr["_y"], rounds=n)):
                m.save_model(str(MODEL_DIR / f"v9_{rep}_{g}_s{s}.txt"))
            print(f"  [{rep}/{g}] {len(tr):,}행 x {n} rounds", flush=True)
        ptr, pcols = pooled_frame(data, cols, pot, mism, ALL_YEARS)
        n = max(int(rounds[f"{rep}|pooled"] * 1.2), 100)
        for s, m in zip(SEEDS, fit_seeds(ptr[pcols], ptr["y_norm"], rounds=n, cat=["gid"])):
            m.save_model(str(MODEL_DIR / f"v9_{rep}_pooled_s{s}.txt"))
        print(f"  [{rep}/pooled] {len(ptr):,}행 x {n} rounds", flush=True)

    (MODEL_DIR / "v9_feature_cols.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "post_params.json").write_text(
        json.dumps({g: {"scale": 1.0, "floor": FLOOR * CAPACITY_KWH[g]} for g in GROUPS},
                   indent=2), encoding="utf-8")
    print("\n학습 완료. python src/inference.py 로 제출 파일을 생성하세요.")


if __name__ == "__main__":
    main()
