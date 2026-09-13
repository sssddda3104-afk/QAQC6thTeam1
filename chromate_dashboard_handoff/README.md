# 크로메이트 정형 데이터 대시보드 전달 패키지

대상은 **1페이지 완료 LOT 조회**와 **2페이지 공정 진행 중 모니터링**입니다. 이미지 모델은 포함하지 않습니다.

## 팀원이 가장 먼저 사용할 파일

### code/dashboard_inference.py
- `current_snapshot(lot_df)` : 현재까지의 표준편차 / 최솟값 / IQR / 표시 기준 이탈률 계산
- `predict_early_warning(lot_df)` : 공정 중 **AI 참고 위험도** 계산
- `predict_final_lot(lot_df)` : 완료 LOT **최종 불량 위험 판정** 계산

### data/dashboard_sensor_timeseries.csv
- 50,094행 Raw 센서 시계열
- 시간 Slider, pH/온도/전압 그래프, Z-score 그래프용

### data/dashboard_progress_checkpoint.csv
- 726 LOT × 10 Progress = 7,260행
- 20/30/40/50/60/65/70/75/80/100% 체크포인트의 누적 Feature
- `EW_OOF_*` : **과거 LOT 화면용 권장 값**. 해당 날짜를 학습에서 제외한 8-fold OOF 점수(seed42)
- `EW_*` : 고정 deployment model의 데모/신규 입력용 점수

### data/dashboard_lot_summary.csv
- 726 LOT 최종 요약
- `Final_OOF_*` : **과거 LOT 화면용 권장 값**. 해당 날짜를 학습에서 제외한 OOF 점수(seed42)
- `Final_*` : 고정 deployment model 데모/신규 입력용 점수

### data/dashboard_daily_summary.csv
- 날짜별 LOT 수, 불량 LOT 수, 양품률, 평균 센서값

## 모델 파일

### 완료 LOT
- `models/final_defect_xgb.json`
- `models/final_preprocess.joblib`
- `models/final_config.json`

검증된 최종 파이프라인: **12 Feature + XGBoost + SMOTE 10%, Target FPR 1%**.
25개 Calibration/Model Seed 조건에서 PR-AUC 0.8037±0.0234, Recall 평균 0.8222, Recall 최소 7/9, 실제 FPR 평균 0.0126.

### Early Warning
- `models/early_warning_xgb.json`
- `models/early_preprocess.joblib`
- `models/early_config.json`

이 모델은 **최종 채택 모델이 아닙니다.** 75~80%에서 신호는 확인됐지만 25-seed 최소 Recall이 6/9로 안정성 기준을 통과하지 못했습니다.
따라서 화면에는 `불량 확률` 대신 **공정 중 AI 참고 위험도**로 표시하세요.

## 과거 데이터 화면과 신규 입력을 구분하세요

- 과거 726 LOT 조회: `Final_OOF_*`, `EW_OOF_*` 사용 권장
  - 이유: 해당 LOT/날짜를 학습에서 제외한 점수라 과거 화면에서도 평가 누수 방지
- 신규 LOT/실시간 입력: `dashboard_inference.py`의 배포 모델 함수 사용

## risk 값 표시
- `risk_percentile` / `*_Risk_Percentile`: 정상 Calibration score 대비 상대 순위
- 실제 불량 발생 확률이 아님
- UI 예시: `AI 참고 위험도 상위 3%` 또는 `정상 기준 위험 순위 97 percentile`

## 표시용 관리 기준과 모델 기준을 구분
- pH 2.20 / Temp 40 / Voltage 15: 대시보드 표시/관리 Rule
- 모델 excursion reference: Development 정상 데이터 5~95% 범위

## 1페이지 연결 추천
- KPI: `dashboard_daily_summary.csv`, `dashboard_lot_summary.csv`
- 과거 LOT 최종 AI 판정: `Final_OOF_Prediction`, `Final_OOF_Risk_Percentile`
- 신규 LOT 판정: `predict_final_lot()`
- 공정 그래프: `dashboard_sensor_timeseries.csv`

## 2페이지 연결 추천
- 시간 Slider / 현재값 / 그래프: `dashboard_sensor_timeseries.csv`
- 시점별 파생변수: `current_snapshot()`
- 과거 LOT 체크포인트 위험도: `EW_OOF_Risk_Percentile`
- 신규 LOT 위험도: `predict_early_warning()`

## 고정 Deployment Calibration 날짜
2021-09-17, 2021-09-23, 2021-09-27, 2021-09-29, 2021-10-12, 2021-10-13, 2021-10-14, 2021-10-18, 2021-10-19, 2021-10-27
