"""학습 스크립트 (대회 규정: 추론 코드와 분리).

최종 레시피 (2024 홀드아웃 검증 총점 0.6438, 기준선 0.5968):
- 피처: LDAPS/GFS 격자·집계 + 발표분 내 컨텍스트(lag/lead/rolling)
- 목적함수: quantile(alpha=0.60) — 평가 필터로 인한 저편향 보정
- 학습 필터: 실제 발전량 < 설비용량 5% 시간대 제외 (평가 대상과 정렬)
- 그룹1/2: 단독 모델, 그룹3: 3그룹 통합(pooled) 학습 (라벨 1년 부족 보완)
- 3시드 앙상블 (42, 202, 777)
- 후처리: 그룹별 (scale, floor) — 홀드아웃에서 튜닝, 연도 간 이전성 검증됨

절차:
1) 2022~23 학습 → 2024 홀드아웃으로 검증 점수·후처리 파라미터·트리 수 산출
2) 전체 데이터 재학습 → models/ 저장 (모델 + post_params.json + feature_cols.txt)
"""
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, TRAIN_DIR
from features import build_features
from metrics import competition_score
from postprocess import apply_post, group_score, tune_post

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
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


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


def _solo_split(data, cols, g, train_years, valid_year):
    d = data.dropna(subset=[g])
    tr = d[d.index.year.isin(train_years)]
    tr = tr[tr[g] >= TRAIN_FILTER_RATIO * CAPACITY_KWH[g]]
    va = d[d.index.year == valid_year]
    return tr[cols], tr[g], (va[cols], va[g]), va[g].to_numpy()


def _pooled_frame(data, cols):
    frames = []
    for g in GROUPS:
        d = data.dropna(subset=[g])
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
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)
    print(f"학습 데이터: {data.shape[0]:,}행, 피처 {len(cols)}개")

    # ── 1) 홀드아웃 검증 + 후처리 튜닝 ──────────────────────
    preds24, actuals24, rounds = {}, {}, {}
    for g in G12:
        tX, ty, va, a = _solo_split(data, cols, g, [2022, 2023], 2024)
        preds24[g], rounds[g] = _fit_predict(tX, ty, va, LGB_PARAMS)
        preds24[g] = np.clip(preds24[g], 0, CAPACITY_KWH[g])
        actuals24[g] = a

    pooled, pcols = _pooled_frame(data, cols)
    ptr = pooled[pooled.index.year <= 2023]
    ptr = ptr[ptr["y_norm"] >= TRAIN_FILTER_RATIO]
    pva = pooled[(pooled.index.year == 2024) & (pooled["gid"] == 3)]
    cap3 = CAPACITY_KWH[G3]
    g3_pred, rounds[G3] = _fit_predict(ptr[pcols], ptr["y_norm"],
                                       (pva[pcols], pva["y_norm"]), LGB_PARAMS,
                                       categorical=["gid"])
    preds24[G3] = np.clip(g3_pred * cap3, 0, cap3)
    actuals24[G3] = (pva["y_norm"] * cap3).to_numpy()

    # 후처리: 그룹1/2는 2023·2024 양년 튜닝 평균(연도 노이즈 헤지), 그룹3은 2024 튜닝
    post = {}
    for g in G12:
        cap = CAPACITY_KWH[g]
        _, sc24, fl24 = tune_post(preds24[g], actuals24[g], cap)
        tX, ty, va23, a23 = _solo_split(data, cols, g, [2022], 2023)
        p23, _ = _fit_predict(tX, ty, va23, LGB_PARAMS)
        _, sc23, fl23 = tune_post(np.clip(p23, 0, cap), a23, cap)
        post[g] = ((sc24 + sc23) / 2, (fl24 + fl23) / 2)
    _, sc3, fl3 = tune_post(preds24[G3], actuals24[G3], cap3)
    post[G3] = (sc3, fl3)

    p_final = {g: apply_post(preds24[g], CAPACITY_KWH[g], *post[g]) for g in GROUPS}
    r = competition_score(p_final, actuals24)
    print("\n===== 2024 홀드아웃 검증 (후처리 포함) =====")
    for g in GROUPS:
        d = r["detail"][g]
        print(f"  {g}: NMAE={d['nmae']:.4f}  FICR={d['ficr']:.4f}  "
              f"post(scale={post[g][0]:.3f}, floor={post[g][1]:.0f}kWh)")
    print(f"  1-NMAE = {r['one_minus_nmae']:.4f}")
    print(f"  FICR   = {r['ficr']:.4f}")
    print(f"  총점   = {r['score']:.4f}")

    # ── 2) 전체 데이터 재학습 및 저장 ──────────────────────
    print("\n전체 데이터로 재학습 중...")
    for g in G12:
        cap = CAPACITY_KWH[g]
        d = data.dropna(subset=[g])
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
