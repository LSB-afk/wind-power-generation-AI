"""대회 공통 상수 및 경로 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submissions"

GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

# 그룹 설비용량 (1시간 기준 kWh 환산값)
CAPACITY_KWH = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}

# 평가 대상: 실제 발전량 >= 설비용량의 10%
EVAL_THRESHOLD_RATIO = 0.10

# 재생에너지 발전량 예측제도 정산 단가 (원/kWh)
# 예측오차율(설비용량 대비 %) <= 6% → 4원, 6~8% → 3원, > 8% → 0원
SETTLEMENT_TIERS = [(6.0, 4.0), (8.0, 3.0)]
MAX_SETTLEMENT_RATE = 4.0

RANDOM_SEED = 42
