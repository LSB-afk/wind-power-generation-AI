"""추론 스크립트 (대회 규정: 학습 코드와 분리).

models/ 에 저장된 LightGBM 모델을 로드해 2025년 8,760시간을 예측하고
submissions/submission.csv 를 생성한다.

평가 데이터는 제출 파일 생성을 위한 추론 목적으로만 사용한다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, DATA_DIR, GROUPS, MODEL_DIR, SUBMISSION_DIR, TEST_DIR
from features import build_features


def main():
    SUBMISSION_DIR.mkdir(exist_ok=True)

    feat = build_features(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")
    feature_cols = (MODEL_DIR / "feature_cols.txt").read_text(encoding="utf-8").splitlines()

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig",
                      parse_dates=["forecast_kst_dtm"])
    X = feat.reindex(sub["forecast_kst_dtm"])[feature_cols]
    missing = X.isna().all(axis=1).sum()
    if missing:
        print(f"경고: 피처가 전혀 없는 시각 {missing}개 (NaN은 LightGBM이 처리)")

    for g in GROUPS:
        model = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_{g}.txt"))
        pred = model.predict(X)
        sub[g] = np.clip(pred, 0, CAPACITY_KWH[g])

    out = SUBMISSION_DIR / "submission.csv"
    # 제출 규격: forecast_id, forecast_kst_dtm 원본 유지, UTF-8
    sub["forecast_kst_dtm"] = sub["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sub.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"제출 파일 생성: {out}")
    print(f"  행 수: {len(sub):,}")
    for g in GROUPS:
        print(f"  {g}: min={sub[g].min():.1f}  mean={sub[g].mean():.1f}  max={sub[g].max():.1f}")


if __name__ == "__main__":
    main()
