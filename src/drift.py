"""2025 테스트 피처 드리프트 점검.

로컬 개선이 Public으로 전이되지 않을 때 가장 먼저 배제해야 할 원인:
테스트 구간에서 일부 피처가 깨졌거나(전량 NaN·상수), 분포가 학습 구간을
크게 벗어나 트리 모델이 외삽 불가 영역으로 나가는 경우.
"""
import numpy as np
import pandas as pd

from config import TEST_DIR, TRAIN_DIR, ROOT
from features import build_features

CACHE = ROOT / "cache"


def psi(train, test, bins=10):
    """Population Stability Index — 학습 분위 기준 테스트 분포 이탈도."""
    tr = train[np.isfinite(train)]
    te = test[np.isfinite(test)]
    if len(tr) < 100 or len(te) < 100:
        return np.nan
    edges = np.unique(np.quantile(tr, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    p = np.histogram(tr, edges)[0] / len(tr) + 1e-6
    q = np.histogram(te, edges)[0] / len(te) + 1e-6
    return float(((q - p) * np.log(q / p)).sum())


def main():
    test_path = CACHE / "test_feat.parquet"
    if test_path.exists():
        test = pd.read_parquet(test_path)
    else:
        print("테스트 피처 생성 중...")
        test = build_features(TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")
        test.to_parquet(test_path)
    train = pd.read_parquet(CACHE / "train_base.parquet")
    from exp_runner import load_cache, add_context
    base, ldaps_raw, _, _ = load_cache()
    davail = ldaps_raw.drop_duplicates("forecast_kst_dtm").set_index(
        "forecast_kst_dtm")["data_available_kst_dtm"]
    train = add_context(base, davail).join(
        pd.read_parquet(CACHE / "train_phys.parquet"), how="left")

    cols = [c for c in train.columns if c in test.columns]
    print(f"공통 피처 {len(cols)}개 | 학습 {len(train):,}행, 테스트 {len(test):,}행")
    missing = set(train.columns) - set(test.columns)
    if missing:
        print(f"!! 테스트에 없는 피처 {len(missing)}개: {sorted(missing)[:5]}")

    rows = []
    for c in cols:
        tr, te = train[c].to_numpy(dtype=float), test[c].to_numpy(dtype=float)
        rows.append({
            "feat": c,
            "nan_tr": np.isnan(tr).mean(),
            "nan_te": np.isnan(te).mean(),
            "mean_tr": np.nanmean(tr), "mean_te": np.nanmean(te),
            "psi": psi(tr, te),
        })
    df = pd.DataFrame(rows)

    broken = df[(df.nan_te > 0.5) & (df.nan_tr < 0.2)]
    print(f"\n[깨진 피처] 테스트 NaN>50% & 학습 NaN<20%: {len(broken)}개")
    if len(broken):
        print(broken[["feat", "nan_tr", "nan_te"]].to_string(index=False))

    const = df[[np.nanstd(test[c].to_numpy(dtype=float)) == 0 for c in cols]]
    print(f"\n[테스트 상수 피처] {len(const)}개" +
          (": " + ", ".join(const.feat.head(10)) if len(const) else ""))

    hi = df.dropna(subset=["psi"]).nlargest(15, "psi")
    print(f"\n[분포 이탈 상위 15] PSI>0.25 = 심각, >0.1 = 주의  (전체 중앙값 {df.psi.median():.3f})")
    for _, r in hi.iterrows():
        flag = "!!" if r.psi > 0.25 else ("!" if r.psi > 0.1 else " ")
        print(f" {flag} {r.feat:44s} PSI {r.psi:.3f}  평균 {r.mean_tr:9.3f} → {r.mean_te:9.3f}")

    print(f"\nPSI>0.25 피처 수: {(df.psi > 0.25).sum()} / {df.psi.notna().sum()}")
    df.to_csv(CACHE / "drift_report.csv", index=False)


if __name__ == "__main__":
    main()
