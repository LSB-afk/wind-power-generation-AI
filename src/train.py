"""학습 스크립트 (대회 규정: 추론 코드와 분리).

v4 레시피 — 이중 폴드(22→23, 22-23→24) 모두에서 이긴 것만 채택:
- 피처: LDAPS/GFS 격자·집계 + 발표분 내 컨텍스트(lag/lead/rolling)
- SCADA 라벨 클리닝: 터빈 정지/계측 불일치 시간대 학습 제외 (양 폴드 +0.010~0.014)
- 목적함수: quantile(alpha=0.60) + 실발전 < 5%cap 학습 제외 (양 폴드 +0.015~0.017)
- 그룹1/2: 단독 모델, 그룹3: 3그룹 통합(pooled) 학습 (fold24 및 H1/H2 일관 +0.009)
- 3시드 앙상블 (42, 202, 777)
- 후처리: floor = 0.10 x 설비용량, scale 없음.
  평가가 '실발전 >= 10%cap' 시간대만 대상이므로 이 하한은 어떤 연도 분포에서도
  채점 시간대를 악화시킬 수 없는 무손실 보정이다. (v2/v3의 공격적 scale/floor는
  2024 튜닝 과적합으로 Public 전이 실패 → 폐기)

절차:
1) 2022~23 학습 → 2024 홀드아웃 검증 점수·트리 수 산출
2) 전체 데이터 재학습 → models/ 저장 (모델 + post_params.json + feature_cols.txt)
"""
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, TRAIN_DIR
from features import build_features
from metrics import competition_score
from postprocess import apply_post
from scada import build_bad_mask

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
TRAIN_FILTER_RATIO = 0.05   # 실제 발전량 < 5%cap 학습 제외
FLOOR_RATIO = 0.10          # 안전 하한 (평가 필터와 동일 — 무손실)
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


def _clean_train(tr: pd.DataFrame, g: str, bad: pd.DataFrame) -> pd.DataFrame:
    """SCADA 나쁜 시간대 + 저발전 시간대 학습 제외."""
    tr = tr[~bad[f"{g}_bad"].reindex(tr.index).fillna(False)]
    return tr


def _fit_predict(tr_X, tr_y, va_X, params, categorical=None):
    """시드 앙상블 학습. 반환: (평균 예측, 시드별 best_iteration 평균)."""
    preds, iters = [], []
    for s in SEEDS:
        p = dict(params, seed=s)
        dtrain = lgb.Dataset(tr_X, tr_y, categorical_feature=categorical or "auto")
        dvalid = lgb.Dataset(va_X[0], va_X[1], reference=dtrain)
        m = lgb.train(p, dtrain, 5000, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(va_X[0], num_iteration=m.best_iteration))
        iters.append(m.best_iteration)
    return np.mean(preds, axis=0), int(np.mean(iters))


def _pooled_frame(data, cols, bad):
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g])
        d = _clean_train(d, g, bad)
        f = d[cols].copy()
        f["y_norm"] = d[g] / CAPACITY_KWH[g]
        f["gid"] = int(g[-1])
        frames.append(f)
    return pd.concat(frames), cols + ["gid"]


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    feat = build_features(TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv")
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"], index_col="kst_dtm")
    bad = build_bad_mask(labels)
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    print(f"학습 데이터: {data.shape[0]:,}행, 피처 {len(cols)}개")
    for g in GROUPS:
        share = (bad[f"{g}_bad"] & data[g].notna()).sum() / data[g].notna().sum()
        print(f"  {g}: SCADA 클리닝 제외 {share:.1%}")

    # ── 1) 2024 홀드아웃 검증 ──────────────────────────────
    preds24, actuals24, rounds = {}, {}, {}
    for g in G12:
        cap = CAPACITY_KWH[g]
        d = data.dropna(subset=[g])
        tr = d[d.index.year <= 2023]
        tr = _clean_train(tr, g, bad)
        tr = tr[tr[g] >= TRAIN_FILTER_RATIO * cap]
        va = d[d.index.year == 2024]
        pred, rounds[g] = _fit_predict(tr[cols], tr[g], (va[cols], va[g]), LGB_PARAMS)
        preds24[g] = np.clip(pred, 0, cap)
        actuals24[g] = va[g].to_numpy()

    pooled, pcols = _pooled_frame(data, cols, bad)
    cap3 = CAPACITY_KWH[G3]
    ptr = pooled[(pooled.index.year <= 2023) & (pooled["y_norm"] >= TRAIN_FILTER_RATIO)]
    # 검증 대상은 클리닝 없이: 원본 2024 그룹3 전체
    va3 = data.dropna(subset=[G3])
    va3 = va3[va3.index.year == 2024]
    va3_X = va3[cols].copy()
    va3_X["gid"] = 3
    g3_pred, rounds[G3] = _fit_predict(ptr[pcols], ptr["y_norm"], (va3_X[pcols], va3[G3] / cap3),
                                       LGB_PARAMS, categorical=["gid"])
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
    for g in G12:
        cap = CAPACITY_KWH[g]
        d = data.dropna(subset=[g])
        d = _clean_train(d, g, bad)
        d = d[d[g] >= TRAIN_FILTER_RATIO * cap]
        n_rounds = max(int(rounds[g] * 1.2), 100)
        for s in SEEDS:
            p = dict(LGB_PARAMS, seed=s)
            m = lgb.train(p, lgb.Dataset(d[cols], d[g]), n_rounds)
            m.save_model(str(MODEL_DIR / f"lgbm_{g}_s{s}.txt"))
        print(f"  [{g}] {len(d):,}행 x {n_rounds} rounds x {len(SEEDS)}시드")

    pall = pooled[pooled["y_norm"] >= TRAIN_FILTER_RATIO]
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
