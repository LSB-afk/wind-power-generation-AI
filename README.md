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

# 1) 전처리: 전 기상변수 x 전 격자 피처 캐시 생성
python src/features_full.py

# 2) 학습: 2024 홀드아웃 검증 후 2022~2024 전체로 27개 모델 학습
python src/train.py

# 3) 추론: 2025년 제출 파일 생성
python src/inference.py
```

검증·진단 재현:

```bash
python src/diagnose.py            # 검증 노이즈 하한·오차 구조 분해
python src/drift.py               # 2025 테스트 피처 드리프트 점검
python src/stage9.py A|B|C|sweep  # v9 후보 학습
python src/stage9.py report       # 페어드 부트스트랩 + 월 부호검정
```

대회 규정에 따라 **학습(train.py)과 추론(inference.py) 코드는 분리**되어 있습니다.

## 현재 접근 방법 (v9 — 2024 홀드아웃 0.6465, Public 0.63853)

v4~v8 이 Public 0.635 에서 정체한 원인을 먼저 진단했다. 7일 블록 부트스트랩 결과
**단일 폴드 총점의 표준오차가 ±0.0063** 으로, 그동안 비교해 온 개선폭(+0.003~5)보다
컸다. 즉 절대점수 비교 자체가 무의미했다. 데이터 버그·외삽 실패·편향 가설은 모두
검정으로 기각했고(상세는 [docs/experiments.md](docs/experiments.md) Stage 11),
남은 병목은 **파워커브 급경사 구간(실발전 25~70%cap, 발전량의 42%)의 정확도**였다.

- **전처리 재설계** ([src/features_full.py](src/features_full.py)): 전 기상변수 x
  전 격자(LDAPS 16 / GFS 9)를 펼쳐 1,577 피처. 기존 파이프라인은 바람 변수만
  격자별로 쓰고 나머지는 평균으로 뭉갰다.
  - **허브고도 외삽 기점 교정**: GFS 는 80/100 m 쌍으로 전단지수를 추정해
    100→117 m 만 외삽(기존은 10 m 기준, 고도비 12배라 alpha 오차가 증폭됐다)
  - 가온도 기반 습윤공기 밀도, IEC 61400-12-1 밀도 보정 풍속, 벌크 리처드슨 수,
    경계층 대비 허브고도, 예보 선행시간, 두 예보원 불일치도
- **지형 보정** ([src/terrain.py](src/terrain.py)): 산악 능선의 가속·차폐는 풍향에
  강하게 의존한다. 학습 기간 SCADA 실측으로 **풍향 12섹터별 승법 보정계수**를
  적합했고(그룹3 은 방향에 따라 0.64~1.12 로 편차가 크다), 보정 풍속을 SCADA
  **경험적 파워커브**에 통과시킨 값을 피처로 넣는다(2단계 파이프라인이 아니라 피처 —
  순수 2단계가 직접 ML 보다 열등하다는 문헌 근거를 따랐다).
- **3표현 앙상블**: base(348) / full(1577) / full+지형(1590) 을 각각 3시드로 학습해
  평균. 세 표현의 우열이 폴드마다 뒤집혀 하나를 고를 근거가 없고, 분포 이동 하에서
  모델 평균이 분산을 줄이기 때문이다(앙상블 SE ±0.0015 < 개별 ±0.0021~0.0023).
- **공통 레시피(v5 계승, 재검증 완료)**: SCADA potential 학습 타깃,
  quantile(alpha=0.60), 저발전 5%cap 학습 제외, 그룹1·2 단독 + 그룹3 통합(pooled),
  후처리는 `floor = 0.10 x 설비용량` 만.

**채택 근거** — 이중 폴드 페어드 블록 부트스트랩 + 월 블록 부호검정:

| | fold24 (22-23→24) | fold23 (22→23) | 월별 승 |
|---|---|---|---|
| v9 vs v8 | +0.0035 ±0.0015, P(개선)=99.0% | +0.0012 ±0.0012, P=85.9% | 18/24 (p≈0.011) |

**규정 준수**: 외부 데이터·사전학습 가중치·외부 API 를 일절 쓰지 않는다. SCADA 는
학습 기간(2022~2024)에만 사용하고, 보정표는 `fit_years` 인자로 적합 연도를 강제해
폴드 간 누수를 막는다. 상세 소명은 [docs/compliance.md](docs/compliance.md).

## 향후 개선 아이디어

v10 에서 앙상블 멤버 추가(MLP·CatBoost·급경사 가중·IDW 표현)를 시도했으나
**전부 게이트 탈락**했다. 멤버는 "비슷하게 강하면서 서로 다를" 때만 도움이 된다
([docs/experiments.md](docs/experiments.md) Stage 12). 남은 방향:

- [ ] 개별 멤버 자체를 강화 (앙상블 확장이 아니라) — 하이퍼파라미터 탐색은 미실시
- [ ] 급경사 구간(25~70%cap) 표적 개선 — 현재 6%밴드 적중률 25%대
- [ ] 월 블록 purged CV 로 폴드 수를 늘려 검정력 확보
- [ ] 격자별·섹터별 파워커브 (현재 그룹 중심 1개)


## 저장소 구조

```
├── src/
│   ├── config.py         # 경로·설비용량·상수
│   ├── features.py       # v8 피처 (base 표현, train/test 공용)
│   ├── features_full.py  # 전 변수 x 전 격자 전처리 (full 표현)
│   ├── terrain.py        # 풍향 섹터 지형 보정 + 경험적 파워커브
│   ├── scada.py          # SCADA 라벨 정제·potential 타깃
│   ├── metrics.py        # NMAE, FICR, 대회 총점
│   ├── postprocess.py    # floor 후처리
│   ├── train.py          # 학습 + 홀드아웃 검증 + 모델 저장
│   ├── inference.py      # 추론 + 제출 파일 생성
│   ├── diagnose.py       # 블록 부트스트랩 노이즈·오차 구조 진단
│   ├── drift.py          # 학습/테스트 분포 드리프트 점검
│   └── stage9.py         # v9 후보 스크리닝 + 유의성 검정
├── docs/
│   ├── experiments.md    # 전체 실험 로그 (Stage 1~11)
│   └── compliance.md     # 대회 규정 준수 소명서
├── models/               # 학습된 LightGBM 모델 + 지형 보정표
├── submissions/          # 제출 CSV
└── notebooks/            # EDA
```
