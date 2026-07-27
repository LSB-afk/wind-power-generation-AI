"""v13 보조 학습 — 의사결정 점예측용 다분위 모델 (base 표현).

v12 앙상블(강한 중심)은 그대로 두고, 조건부 분포에서 계산한 **이동량만** 얹기 위해
base 표현으로 분위 11개를 학습한다. 이동량 = (기대점수 최적점 − q60).

검증 결과: v12 대비 fold24 −0.0002(노이즈 수준), fold23 +0.0029(P=99.0%, 2SE 통과).
양 폴드 동시 양수라는 정식 게이트는 통과하지 못했으므로 **주 제출물이 아니라
별도 후보**로 생성한다. 리더보드가 최고점 파일을 유지하므로 실측 판정이 안전하다.

산출물: models/v13_q{tau}_{family}.txt, submissions/submission_v13_decision.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from config import CAPACITY_KWH, GROUPS, MODEL_DIR, ROOT, TRAIN_DIR
from exp_runner import BASE_PARAMS, add_context, load_cache
from scada import build_mismatch_mask, build_potential
from stage13 import TAUS, ROUNDS

CACHE = ROOT / "cache"
ALL_YEARS = [2022, 2023, 2024]
FILTER = 0.05
G12 = ["kpx_group_1", "kpx_group_2"]
G3 = "kpx_group_3"


def c7(alpha, seed=42):
    return dict(BASE_PARAMS, objective="quantile", alpha=alpha, seed=seed,
                learning_rate=0.02, num_leaves=63, min_data_in_leaf=100,
                feature_fraction=0.20, bagging_fraction=0.7, bagging_freq=1,
                lambda_l2=30.0)


def base_features():
    base, ldaps_raw, _, _ = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    return add_context(base, davail).join(
        pd.read_parquet(CACHE / "train_phys.parquet"), how="left")


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    feat = base_features()
    labels = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig",
                         parse_dates=["kst_dtm"], index_col="kst_dtm")
    pot, mism = build_potential(labels), build_mismatch_mask(labels)
    data = feat.join(labels, how="inner")
    cols = list(feat.columns)

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
    pooled = pooled[pooled.index.year.isin(ALL_YEARS) & (pooled["y_norm"] >= FILTER)]
    pcols = cols + ["gid"]

    # 기대 점수 계산에 쓸 ā (학습 연도 평가 대상 평균 발전량)
    abar = {}
    for g in GROUPS:
        lab = data[g].dropna()
        lab = lab[lab.index.year.isin(ALL_YEARS)]
        abar[g] = float(lab[lab >= 0.10 * CAPACITY_KWH[g]].mean())

    for g in G12:
        cap = CAPACITY_KWH[g]
        d = data.dropna(subset=[g]).copy()
        d["_y"] = pot[f"{g}_potential"].reindex(d.index)
        d = d[~mism[f"{g}_mismatch"].reindex(d.index).fillna(False)].dropna(subset=["_y"])
        d = d[d.index.year.isin(ALL_YEARS) & (d["_y"] >= FILTER * cap)]
        for a in TAUS:
            m = lgb.train(c7(a), lgb.Dataset(d[cols], d["_y"] / cap), ROUNDS)
            m.save_model(str(MODEL_DIR / f"v13_q{int(a*100):02d}_{g}.txt"))
        print(f"  [{g}] {len(d):,}행 x 분위 {len(TAUS)}개", flush=True)

    for a in TAUS:
        m = lgb.train(c7(a), lgb.Dataset(pooled[pcols], pooled["y_norm"],
                                         categorical_feature=["gid"]), ROUNDS)
        m.save_model(str(MODEL_DIR / f"v13_q{int(a*100):02d}_pooled.txt"))
    print(f"  [pooled] {len(pooled):,}행 x 분위 {len(TAUS)}개")

    np.savez(MODEL_DIR / "v13_meta.npz", taus=TAUS,
             **{f"abar_{g}": np.array([abar[g]]) for g in GROUPS},
             cols=np.array(cols, dtype=object))
    print("\nv13 분위 모델 저장 완료. python src/inference_decision.py 로 후보 제출물 생성.")


if __name__ == "__main__":
    main()
