"""v13 후보 제출물 — v12 앙상블 중심 + 의사결정 이론 이동량.

v12 추론 결과(submissions/submission.csv)를 중심으로 두고, base 표현 분위 모델로
2025년 조건부 분포를 세워 (기대점수 최적점 − q60) 만큼 이동시킨다.

정식 게이트(양 폴드 동시 양수)는 통과하지 못한 후보이므로 주 제출물을 덮어쓰지 않고
submissions/submission_v13_decision.csv 로 따로 저장한다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, DATA_DIR, GROUPS, MODEL_DIR, SUBMISSION_DIR, TEST_DIR
from decision import optimal_point
from features import build_features

G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"
FLOOR = 0.10


def main():
    meta = np.load(MODEL_DIR / "v13_meta.npz", allow_pickle=True)
    taus = meta["taus"]
    cols = list(meta["cols"])

    v12 = pd.read_csv(SUBMISSION_DIR / "submission.csv", encoding="utf-8-sig",
                      parse_dates=["forecast_kst_dtm"])
    feat = build_features(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")
    X = feat.reindex(v12["forecast_kst_dtm"])[cols]
    q60_idx = int(np.argmin(np.abs(taus - 0.60)))

    out = v12.copy()
    for g in GROUPS:
        cap = CAPACITY_KWH[g]
        fam = "pooled" if g == G3 else g
        XX = X
        if g == G3:
            XX = X.copy()
            XX["gid"] = 3
        Q = []
        for a in taus:
            m = lgb.Booster(model_file=str(MODEL_DIR / f"v13_q{int(a*100):02d}_{fam}.txt"))
            Q.append(m.predict(XX) * cap)
        Q = np.clip(np.stack(Q, axis=1), 0, cap)
        abar = float(meta[f"abar_{g}"][0])
        shift = optimal_point(Q, taus, cap, abar) - Q[:, q60_idx]
        out[g] = np.clip(np.maximum(v12[g].to_numpy() + shift, FLOOR * cap), 0, cap)
        print(f"  {g}: 평균 이동 {shift.mean()/cap:+.4f}cap")

    path = SUBMISSION_DIR / "submission_v13_decision.csv"
    out["forecast_kst_dtm"] = out["forecast_kst_dtm"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n후보 제출 파일: {path}\n  행 수: {len(out):,}")
    for g in GROUPS:
        print(f"  {g}: min={out[g].min():.1f}  mean={out[g].mean():.1f}  max={out[g].max():.1f}")


if __name__ == "__main__":
    main()
