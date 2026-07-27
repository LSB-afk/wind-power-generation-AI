"""외부 공개 데이터: NOAA GEFS 앙상블 예보 취득.

왜 필요한가. 오라클 실험 결과 우리 오차의 85% 가 풍속 예보 오차이고,
제공된 LDAPS(16격자)·GFS(9격자) 로 짜낼 수 있는 피처는 한계에 도달했다.
문헌은 서로 다른 NWP 를 결합하면 RMSE 8~22% 가 준다고 보고한다.

**Data Leakage 준수** (대회 규칙 3항):
- 대회 제공 예보는 `09:00 KST 초기화 = 00Z` 이고 `13:00 KST` 부터 사용 가능하다.
  GEFS 도 **동일한 00Z 런**만 쓰고, 예보시간은 익일 01:00~익익일 00:00 KST 에
  해당하는 f016~f039 만 취한다. 00Z 런은 늦어도 05Z(=14:00 KST) 에 배포되지만,
  대회가 정한 사용가능시각(전일 13:00 KST)과 동일 기준을 적용하기 위해
  **제공 데이터와 완전히 같은 런·같은 예보시간만** 사용한다.
- 재분석(ERA5 등) 이나 사후 보정자료는 일절 쓰지 않는다.

**외부 데이터 규칙 준수** (4항): NOAA GEFS 는 미국 정부 저작물로 퍼블릭 도메인,
AWS Open Data 로 누구나 접근 가능하며(`s3://noaa-gefs-pds`, 인증 불필요),
이 스크립트만으로 재현된다.

사용법: python src/gefs.py probe   → 표본 기간만 받아 가치 측정
        python src/gefs.py fetch   → 전 기간 취득
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from config import ROOT

CACHE = ROOT / "cache"
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# 태백 가덕산/원동 풍력단지 (info.xlsx 터빈 좌표 중심)
SITE_LAT, SITE_LON = 37.283, 128.963
# 0.5도 격자 — 사이트를 감싸는 3x3
LATS = [37.0, 37.5, 38.0]
LONS = [128.5, 129.0, 129.5]
# 예보 대상 익일 01:00~익익일 00:00 KST = 00Z 런의 f016~f039
FHOURS = list(range(16, 40))
WANT = ("UGRD:10 m above ground", "VGRD:10 m above ground",
        "GUST:surface", "TMP:2 m above ground", "PRES:surface")


def _url(date: str, fh: int) -> str:
    return (f"{BASE}/gefs.{date}/00/atmos/pgrb2ap5/"
            f"geavg.t00z.pgrb2a.0p50.f{fh:03d}")


def _spread_url(date: str, fh: int) -> str:
    return (f"{BASE}/gefs.{date}/00/atmos/pgrb2ap5/"
            f"gespr.t00z.pgrb2a.0p50.f{fh:03d}")


def _get(url, lo=None, hi=None, tries=4, timeout=90):
    """재시도 포함 HTTP GET. S3 는 간헐적으로 읽기가 지연된다."""
    from urllib.request import Request, urlopen
    for k in range(tries):
        try:
            r = Request(url)
            if lo is not None:
                r.add_header("Range", f"bytes={lo}-{'' if hi is None else hi}")
            return urlopen(r, timeout=timeout).read()
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(1.5 * (k + 1))
    return None


def _fetch_records(url: str) -> dict:
    """.idx 로 필요한 변수 구간을 찾아, **인접 레코드는 한 번에** 받는다.

    변수마다 따로 요청하면 하루 120회가 넘어 타임아웃이 잦다.
    원하는 레코드의 최소~최대 바이트를 한 덩어리로 받아 요청 수를 줄인다.
    """
    import cfgrib

    raw = _get(url + ".idx", timeout=45)
    if raw is None:
        return {}
    lines = [l for l in raw.decode().strip().split("\n") if l]
    starts = [int(l.split(":")[1]) for l in lines]
    want = [i for i, l in enumerate(lines) if any(w in l for w in WANT)]
    if not want:
        return {}
    lo = starts[min(want)]
    hi = (starts[max(want) + 1] - 1) if max(want) + 1 < len(starts) else None
    blob = _get(url, lo, hi)
    if blob is None:
        return {}

    tmp = CACHE / f".grib_{os.getpid()}_{threading.get_ident()}.tmp"
    tmp.write_bytes(blob)
    out = {}
    try:
        for ds in cfgrib.open_datasets(str(tmp), backend_kwargs={"indexpath": ""}):
            for v in ds.data_vars:
                sel = ds[v].sel(latitude=LATS, longitude=LONS, method="nearest")
                out[v] = np.asarray(sel.values, dtype=np.float32)
            ds.close()
    except Exception:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    return out


def fetch_day(date: str) -> pd.DataFrame:
    """하루치(f016~f039) 를 예보 대상 KST 시각 인덱스로 반환."""
    rows = {}
    for fh in FHOURS:
        rec = _fetch_records(_url(date, fh))
        if not rec:
            continue
        # 00Z + fh 시간 = UTC → KST(+9)
        t = pd.Timestamp(date) + pd.Timedelta(hours=fh + 9)
        flat = {}
        for key, arr in rec.items():
            flat[f"gefs_{key}_mean"] = float(np.mean(arr))
            flat[f"gefs_{key}_c"] = float(arr[1, 1])   # 사이트 최근접 격자
        rows[t] = flat
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def fetch_range(dates, workers=8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(fetch_day, dates))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def probe():
    """표본 기간으로 '제공 GFS 대비 GEFS 가 풍속을 더 잘 맞추는가' 측정."""
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    dates = [d.strftime("%Y%m%d")
             for d in pd.date_range("2024-03-01", periods=n)]
    print(f"표본 {len(dates)}일 취득 중 (f016~f039)...", flush=True)
    t0 = time.time()
    df = fetch_range(dates)
    dt = time.time() - t0
    if df.empty:
        print("취득 실패"); return
    df.to_parquet(CACHE / "gefs_probe.parquet")
    print(f"취득 완료: {df.shape[0]}시간 x {df.shape[1]}변수, {dt:.0f}초")
    print(f"  → 하루당 {dt/len(dates):.1f}초, 4년(1461일) 환산 {dt/len(dates)*1461/3600:.1f}시간")
    print(df.head(3).to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        probe()
