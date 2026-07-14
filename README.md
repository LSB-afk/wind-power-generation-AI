# 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026

기상청 기상예보 데이터(LDAPS, GFS)를 활용해 태백 가덕산/원동 풍력단지 3개 KPX 그룹의
시간별 발전량(kWh)을 예측하는 대회 워크스페이스입니다.

- 대회 링크: https://dacon.io/competitions/official/236727/overview/description
- 대회 기간: 2026.07.06 ~ 2026.08.14 (1차 제출 마감 08.14 10:00)

## 예측 대상

| 그룹 | 터빈 | 설비용량 | 1시간 환산 |
|---|---|---:|---:|
| `kpx_group_1` | VESTAS V126 1~6호기 | 21.6 MW | 21,600 kWh |
| `kpx_group_2` | VESTAS V126 7~12호기 | 21.6 MW | 21,600 kWh |
| `kpx_group_3` | UNISON U136 1~5호기 | 21.0 MW | 21,000 kWh |

- 위치: 강원 태백 가덕산/원동 (허브고도 117 m)
- 예측 기간: 2025-01-01 01:00 ~ 2026-01-01 00:00 (8,760시간)

## 평가 산식

`총점 = 0.5 × (1 - NMAE) + 0.5 × FICR`

- 실제 발전량이 **설비용량의 10% 이상**인 시간대만 평가
- NMAE: `mean(|예측 - 실제| / 그룹 설비용량)` 의 3그룹 평균
- FICR(정산금획득률): 시간별 예측오차율(설비용량 대비 %) 구간별 정산단가를 적용한
  획득 정산금 / 이론상 최대 정산금 (`src/metrics.py` 참고)

## 데이터 배치

대회 데이터는 재배포 금지이므로 git에 포함하지 않습니다.
DACON에서 다운로드 후 아래 구조로 배치하세요.

```
Data/
├── info.xlsx
├── data_description.md
├── sample_submission.csv
├── train/
│   ├── ldaps_train.csv      # LDAPS 예보 (16격자, 2022~2024)
│   ├── gfs_train.csv        # GFS 예보 (9격자, 2022~2024)
│   ├── train_labels.csv     # 그룹별 실제 발전량 (그룹3은 2023년부터)
│   ├── scada_vestas_train.csv
│   └── scada_unison_train.csv
└── test/
    ├── ldaps_test.csv       # 2025년 LDAPS 예보
    └── gfs_test.csv         # 2025년 GFS 예보
```

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 승격 검증: v5와 v6를 3시드·이중 폴드로 비교
python src/exp_runner.py stage7

# 2) 학습: PASS 게이트를 검증한 뒤 2022~2024 전체 데이터로 9개 모델 학습
python src/train.py

# 3) 추론: 모델·레시피·입력 해시 검증 후 2025년 제출 파일 생성
python src/inference.py
```

대회 규정에 따라 **학습(train.py)과 추론(inference.py) 코드는 분리**되어 있습니다.

## 현재 접근 방법 (v6 — 터빈 가중 potential, 3시드 이중 폴드 PASS)

Public `0.63595`를 만든 v4 이후 **이중 폴드(22→23, 22-23→24) 모두에서 이기는
변경만 채택**한다. v6는 v5 potential 타깃의 동일 터빈 가정을 학습 연도 SCADA에서
추정한 터빈별 상대 출력 비중으로 교정한다. 검증 연도 SCADA는 보정치나 학습 타깃에
사용하지 않는다.

| 폴드 | v5 3시드 | v6 weighted 3시드 | 차이 |
|---|---:|---:|---:|
| fold23 | 0.632360740 | **0.634387247** | **+0.002026507** |
| fold24 | 0.638295889 | **0.638535175** | **+0.000239286** |
| 평균 개선 | | | **+0.001132897** |

- 최종 제출: `submissions/submission.csv`
- 제출 SHA-256: `2e763e97b53ea0698a2a7986635333fa0b0ef3bb8e5cc77726aa64638d5f6bf2`
- 생산 레시피 SHA-256: `94fd21bcddb9e9a88945d529bd2000e265468ad92ef247372603252e8d10eba6`
- 제출 규격: 8,760행 × 5열, 샘플 ID·시간 순서 일치, NaN/Inf 없음, 설비용량 범위 준수

위 수치는 로컬 시계열 검증 결과이며, Public 점수 개선은 DACON 제출 후에만 확인할 수 있다.

- **피처 (276개)**: LDAPS 16격자 / GFS 9격자의 풍속(u,v → 속력·풍향), 허브고도 인접
  (GFS 80/100 m) 풍속, 돌풍, 공기밀도 근사, 경계층 높이, 시간 주기성(sin/cos)
  + **발표분 내 컨텍스트**(같은 data_available 24시간 블록 안의 lag/lead/rolling — 누수 없음)
- **SCADA 가중 potential 타깃** (`src/scada.py`): 정상 터빈 출력으로 결측·정지 터빈의
  잠재 출력을 복원하되, 학습 연도에서 추정한 터빈별 상대 출력 비중을 적용한다.
  라벨 연도는 `raw_timestamp.ceil("h")`로 판정해 연말 경계 누수를 차단한다.
- **모델**: LightGBM **quantile(alpha=0.60)** — 평가 필터(실발전 ≥ 설비용량 10%)로 인한
  저편향을 목적함수 수준에서 보정. 실제 발전량 < 5%cap 시간대는 학습에서 제외.
- **그룹3 통합학습**: 그룹3은 라벨이 2023년부터라 데이터가 절반 → 3그룹을 정규화 타깃
  (y/설비용량) + 그룹ID로 통합 학습
- **3시드 앙상블**: seed 42/202/777 평균
- **후처리**: `floor = 0.10 × 설비용량`만 적용. 평가 시간대는 정의상 실발전 ≥ 10%cap이므로
  이 하한은 어떤 연도 분포에서도 채점 시간대를 악화시킬 수 없는 **무손실 보정**.
  (v2/v3의 2024 튜닝 scale/floor는 Public 전이 실패로 폐기)
- **누수 방지**: 제공 예보 데이터는 전일 13:00(KST) 발표분만 포함되어 있어
  예측기준시점 규칙을 자동 준수. 컨텍스트 피처도 발표분 블록 내부로 제한.
  SCADA는 학습 타깃 생성에만 사용. 2025년 실측/사후 자료는 일절 미사용.

실험 과정 전체는 [docs/experiments.md](docs/experiments.md) 참고.
최종 승격 재현: `python src/exp_runner.py stage7`

## 향후 개선 아이디어

- [x] 학습 연도 SCADA에서 터빈별 상대 출력 비중을 추정한 weighted potential 타깃
- [ ] LDAPS/GFS 두 예보원 불일치도(스프레드) 피처 → 예보 불확실성 반영
- [ ] 후처리 고도화: floor를 예측 구간별 조건부 기대값으로 대체
- [ ] CatBoost/XGBoost 이종 앙상블
- [ ] 외부 공개 데이터 (ASOS 관측 등 — 예측기준시점 규칙 준수 범위 내)

## 저장소 구조

```
├── src/
│   ├── config.py      # 경로·설비용량·상수
│   ├── features.py    # LDAPS/GFS 피처 엔지니어링 (train/test 공용)
│   ├── metrics.py     # NMAE, FICR, 대회 총점
│   ├── train.py       # 학습 + 검증 + 모델 저장
│   └── inference.py   # 추론 + 제출 파일 생성
├── models/            # 학습된 LightGBM 모델 (.txt)
├── submissions/       # 제출 CSV
└── notebooks/         # EDA
```
