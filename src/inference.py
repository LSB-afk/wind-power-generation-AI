"""추론 스크립트 (대회 규정: 학습 코드와 분리).

models/ 의 v9 앙상블(3개 피처 표현 x 3시드, 그룹1·2 단독 + 그룹3 통합)을 로드해
2025년 8,760시간을 예측하고 floor 후처리를 적용한 뒤 제출 CSV를 생성한다.

평가 데이터는 제출 파일 생성을 위한 추론 목적으로만 사용한다.
지형 보정표는 학습 단계에서 2022~2024 SCADA로 적합해 저장된 것을 읽기만 하며,
2025 구간의 실측은 사용하지 않는다.
"""
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

import features_full
from config import CAPACITY_KWH, DATA_DIR, GROUPS, MODEL_DIR, ROOT, SUBMISSION_DIR, TEST_DIR
from features import build_features
from postprocess import apply_post
from smooth import smooth_within_block
from terrain import apply_terrain, load_terrain

CACHE = ROOT / "cache"
REPS = ("base", "full", "fullterr")
SEEDS = (42, 202, 777)
SMOOTH_HOURS = 3
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


def build_test_reps():
    """추론용 3개 표현 — 학습과 동일한 전처리 경로를 탄다."""
    base = build_features(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")
    full_path = CACHE / "full_test.parquet"
    full = (pd.read_parquet(full_path) if full_path.exists()
            else features_full.build(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv"))
    tparams = load_terrain(MODEL_DIR / "terrain_params.npz")
    return {"base": base, "full": full,
            "fullterr": full.join(apply_terrain(full, tparams), how="left")}


def ensemble(prefix, X):
    return np.mean([lgb.Booster(model_file=str(MODEL_DIR / f"{prefix}_s{s}.txt")).predict(X)
                    for s in SEEDS], axis=0)


def main():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    reps = build_test_reps()
    cols = json.loads((MODEL_DIR / "v9_feature_cols.json").read_text(encoding="utf-8"))
    post = json.loads((MODEL_DIR / "post_params.json").read_text(encoding="utf-8"))

    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig",
                      parse_dates=["forecast_kst_dtm"])
    t = sub["forecast_kst_dtm"]

    preds = {g: [] for g in GROUPS}
    for rep in REPS:
        X = reps[rep].reindex(t)[cols[rep]]
        missing = X.isna().all(axis=1).sum()
        if missing:
            print(f"경고: {rep} 피처가 전혀 없는 시각 {missing}개 (NaN은 LightGBM이 처리)")
        for g in G12:
            cap = CAPACITY_KWH[g]
            preds[g].append(np.clip(ensemble(f"v9_{rep}_{g}", X), 0, cap))
        cap3 = CAPACITY_KWH[G3]
        Xp = X.copy()
        Xp["gid"] = 3
        preds[G3].append(np.clip(ensemble(f"v9_{rep}_pooled", Xp) * cap3, 0, cap3))
        print(f"  {rep}: 추론 완료")

    # 발표분 블록 내 3시간 평활 — 시간별 독립 예측의 잡음 제거.
    # 블록(01:00~익일 00:00) 밖과는 절대 섞지 않으므로 예측기준시점 규칙을 지킨다.
    # 이중 폴드 검증: fold24 +0.0011, fold23 +0.0013 (FICR 양쪽 상승)
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        pred = apply_post(np.mean(preds[g], axis=0), cap,
                          post[g]["scale"], post[g]["floor"])
        pred = smooth_within_block(pred, pd.DatetimeIndex(t), SMOOTH_HOURS)
        sub[g] = np.clip(np.maximum(pred, post[g]["floor"]), 0, cap)

    out = SUBMISSION_DIR / "submission.csv"
    # 제출 규격: forecast_id, forecast_kst_dtm 원본 유지, UTF-8
    sub["forecast_kst_dtm"] = sub["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sub.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n제출 파일 생성: {out}\n  행 수: {len(sub):,}")
    for g in GROUPS:
        print(f"  {g}: min={sub[g].min():.1f}  mean={sub[g].mean():.1f}  max={sub[g].max():.1f}")


if __name__ == "__main__":
    main()
