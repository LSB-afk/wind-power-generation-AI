"""학습 스크립트 (대회 규정: 추론 코드와 분리).

1) 2022~2023 학습 → 2024 홀드아웃 검증으로 대회 총점 확인
   (테스트와 동일한 '연 단위 미래 예측' 구조 재현)
2) 전체 학습 데이터로 재학습 → models/ 에 저장
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, RANDOM_SEED, TRAIN_DIR
from features import build_features
from metrics import competition_score

LGB_PARAMS = {
    "objective": "l1",          # MAE — NMAE 직접 최적화
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
NUM_ROUNDS = 5000
EARLY_STOP = 200


def load_train_data():
    feat = build_features(TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv")
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"], index_col="kst_dtm")
    return feat.join(labels, how="inner"), [c for c in feat.columns]


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    data, feature_cols = load_train_data()
    print(f"학습 데이터: {data.shape[0]:,}행, 피처 {len(feature_cols)}개")

    # ── 1) 2024 홀드아웃 검증 ──────────────────────────────
    preds, actuals, best_iters = {}, {}, {}
    for g in GROUPS:
        d = data.dropna(subset=[g])
        tr = d[d.index.year <= 2023]
        va = d[d.index.year == 2024]
        dtrain = lgb.Dataset(tr[feature_cols], tr[g])
        dvalid = lgb.Dataset(va[feature_cols], va[g], reference=dtrain)
        model = lgb.train(LGB_PARAMS, dtrain, NUM_ROUNDS, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best_iters[g] = model.best_iteration
        pred = np.clip(model.predict(va[feature_cols]), 0, CAPACITY_KWH[g])
        preds[g], actuals[g] = pred, va[g].to_numpy()
        print(f"[{g}] train {len(tr):,} / valid {len(va):,} / best_iter {model.best_iteration}")

    result = competition_score(preds, actuals)
    print("\n===== 2024 홀드아웃 검증 결과 =====")
    for g in GROUPS:
        d = result["detail"][g]
        print(f"  {g}: NMAE={d['nmae']:.4f}  FICR={d['ficr']:.4f}")
    print(f"  1-NMAE = {result['one_minus_nmae']:.4f}")
    print(f"  FICR   = {result['ficr']:.4f}")
    print(f"  총점   = {result['score']:.4f}")

    # ── 2) 전체 데이터 재학습 및 저장 ──────────────────────
    print("\n전체 데이터로 재학습 중...")
    for g in GROUPS:
        d = data.dropna(subset=[g])
        # 홀드아웃 best_iteration을 학습량 증가(데이터 ~1.5배)에 맞춰 보정
        n_rounds = max(int(best_iters[g] * 1.2), 100)
        dtrain = lgb.Dataset(d[feature_cols], d[g])
        model = lgb.train(LGB_PARAMS, dtrain, n_rounds)
        out = MODEL_DIR / f"lgbm_{g}.txt"
        model.save_model(str(out))
        print(f"  [{g}] {len(d):,}행 x {n_rounds} rounds → {out}")

    # 추론 시 동일 피처 순서 보장
    (MODEL_DIR / "feature_cols.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    print("\n학습 완료. python src/inference.py 로 제출 파일을 생성하세요.")


if __name__ == "__main__":
    main()
