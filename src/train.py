"""학습 스크립트 (대회 규정: 추론 코드와 분리).

v5 레시피 — 이중 폴드(22→23, 22-23→24) 모두에서 이긴 것만 채택:
- 피처: LDAPS/GFS 격자·집계 + 발표분 내 컨텍스트(lag/lead/rolling)
- 학습 타깃: SCADA 기반 **potential**(가용률 보정 발전량) — 정지 터빈 시간대를
  버리는 대신 '정상 터빈 평균 x 전체 대수'로 복원 (v4 대비 fold23 +0.009, fold24 +0.003)
  가용률 계수 역보정은 이중 폴드에서 해로워 미적용 (quantile 상향과 상쇄됨)
- 라벨-SCADA 계측 불일치(>5%cap) 시간대만 학습 제외
- 목적함수: quantile(alpha=0.60) + potential < 5%cap 학습 제외
- 그룹1/2: 단독 모델, 그룹3: 3그룹 통합(pooled) 학습
- 3시드 앙상블 (42, 202, 777)
- 후처리: floor = 0.10 x 설비용량만 (평가 필터 정의상 무손실). scale 없음.

검증은 항상 실제 라벨(KPX 발전량) 기준. potential은 학습 타깃으로만 사용.
"""
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, TRAIN_DIR
from features import build_features
from metrics import competition_score
from postprocess import apply_post
from scada import build_mismatch_mask, build_potential

LGB_PARAMS = {
    "objective": "quantile",
    "alpha": 0.60,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}
SEEDS = (42, 202, 777)
TRAIN_FILTER_RATIO = 0.05   # potential < 5%cap 학습 제외
FLOOR_RATIO = 0.10          # 안전 하한 (평가 필터와 동일 — 무손실)
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


def _solo_train_frame(data, cols, g, pot, mism, years):
    """그룹 단독 학습 프레임: potential 타깃, 불일치/저발전 제외."""
    cap = CAPACITY_KWH[g]
    d = data.dropna(subset=[g])
    tr = d[d.index.year.isin(years)].copy()
    tr["_y"] = pot[f"{g}_potential"].reindex(tr.index)
    tr = tr[~mism[f"{g}_mismatch"].reindex(tr.index).fillna(False)]
    tr = tr.dropna(subset=["_y"])
    return tr[tr["_y"] >= TRAIN_FILTER_RATIO * cap]


def _pooled_train_frame(data, cols, pot, mism, years):
    """통합 학습 프레임: 정규화 potential 타깃 + 그룹ID."""
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g]).copy()
        d["_y"] = pot[f"{g}_potential"].reindex(d.index)
        d = d[~mism[f"{g}_mismatch"].reindex(d.index).fillna(False)]
        d = d.dropna(subset=["_y"])
        f = d[cols].copy()
        f["y_norm"] = d["_y"] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        frames.append(f)
    pooled = pd.concat(frames)
    pooled = pooled[pooled.index.year.isin(years)]
    return pooled[pooled["y_norm"] >= TRAIN_FILTER_RATIO], cols + ["gid"]


def _fit_predict(tr_X, tr_y, va_X, va_y, params, categorical=None):
    """시드 앙상블 학습. 반환: (평균 예측, 시드별 best_iteration 평균)."""
    preds, iters = [], []
    for s in SEEDS:
        p = dict(params, seed=s)
        dtrain = lgb.Dataset(tr_X, tr_y, categorical_feature=categorical or "auto")
        dvalid = lgb.Dataset(va_X, va_y, reference=dtrain)
        m = lgb.train(p, dtrain, 5000, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(va_X, num_iteration=m.best_iteration))
        iters.append(m.best_iteration)
    return np.mean(preds, axis=0), int(np.mean(iters))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    feat = build_features(TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv")
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"], index_col="kst_dtm")
    pot = build_potential(labels)
    mism = build_mismatch_mask(labels)
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    print(f"학습 데이터: {data.shape[0]:,}행, 피처 {len(cols)}개")

    # ── 1) 2024 홀드아웃 검증 (실제 라벨 기준) ──────────────
    preds24, actuals24, rounds = {}, {}, {}
    for g in G12:
        cap = CAPACITY_KWH[g]
        tr = _solo_train_frame(data, cols, g, pot, mism, [2022, 2023])
        d = data.dropna(subset=[g])
        va = d[d.index.year == 2024]
        pred, rounds[g] = _fit_predict(tr[cols], tr["_y"], va[cols], va[g], LGB_PARAMS)
        preds24[g] = np.clip(pred, 0, cap)
        actuals24[g] = va[g].to_numpy()

    ptr, pcols = _pooled_train_frame(data, cols, pot, mism, [2022, 2023])
    cap3 = CAPACITY_KWH[G3]
    d3 = data.dropna(subset=[G3])
    va3 = d3[d3.index.year == 2024]
    va3_X = va3[cols].copy()
    va3_X["gid"] = 3
    g3_pred, rounds[G3] = _fit_predict(ptr[pcols], ptr["y_norm"], va3_X[pcols],
                                       va3[G3] / cap3, LGB_PARAMS, categorical=["gid"])
    preds24[G3] = np.clip(g3_pred * cap3, 0, cap3)
    actuals24[G3] = va3[G3].to_numpy()

    post = {g: (1.0, FLOOR_RATIO * CAPACITY_KWH[g]) for g in GROUPS}
    p_final = {g: apply_post(preds24[g], CAPACITY_KWH[g], *post[g]) for g in GROUPS}
    r = competition_score(p_final, actuals24)
    print("\n===== 2024 홀드아웃 검증 (floor10 포함) =====")
    for g in GROUPS:
        d = r["detail"][g]
        print(f"  {g}: NMAE={d['nmae']:.4f}  FICR={d['ficr']:.4f}")
    print(f"  1-NMAE = {r['one_minus_nmae']:.4f}")
    print(f"  FICR   = {r['ficr']:.4f}")
    print(f"  총점   = {r['score']:.4f}")

    # ── 2) 전체 데이터 재학습 및 저장 ──────────────────────
    print("\n전체 데이터로 재학습 중...")
    all_years = [2022, 2023, 2024]
    for g in G12:
        tr = _solo_train_frame(data, cols, g, pot, mism, all_years)
        n_rounds = max(int(rounds[g] * 1.2), 100)
        for s in SEEDS:
            p = dict(LGB_PARAMS, seed=s)
            m = lgb.train(p, lgb.Dataset(tr[cols], tr["_y"]), n_rounds)
            m.save_model(str(MODEL_DIR / f"lgbm_{g}_s{s}.txt"))
        print(f"  [{g}] {len(tr):,}행 x {n_rounds} rounds x {len(SEEDS)}시드")

    pall, pcols = _pooled_train_frame(data, cols, pot, mism, all_years)
    n_rounds = max(int(rounds[G3] * 1.2), 100)
    for s in SEEDS:
        p = dict(LGB_PARAMS, seed=s)
        m = lgb.train(p, lgb.Dataset(pall[pcols], pall["y_norm"],
                                     categorical_feature=["gid"]), n_rounds)
        m.save_model(str(MODEL_DIR / f"lgbm_pooled_s{s}.txt"))
    print(f"  [pooled(그룹3용)] {len(pall):,}행 x {n_rounds} rounds x {len(SEEDS)}시드")

    (MODEL_DIR / "feature_cols.txt").write_text("\n".join(cols), encoding="utf-8")
    (MODEL_DIR / "post_params.json").write_text(
        json.dumps({g: {"scale": post[g][0], "floor": post[g][1]} for g in GROUPS},
                   indent=2), encoding="utf-8")
    print("\n학습 완료. python src/inference.py 로 제출 파일을 생성하세요.")


if __name__ == "__main__":
    main()
