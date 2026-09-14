"""
팀원이 전달한 실제 모델 예측 결과(handoff 패키지) 로딩 함수.

과거 726 LOT 화면에는 반드시 OOF(Out-Of-Fold) 점수를 쓴다 — 해당 LOT/날짜를
학습에서 제외하고 낸 점수라 과거 화면에서도 평가 누수가 없다 (README 권고).
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
import shap

BASE = Path(__file__).parent / "handoff_data"
SHAP_MODEL_DIR = Path(__file__).parent / "chromate_dashboard_handoff" / "models"

PROGRESS_LEVELS = [20, 30, 40, 50, 60, 65, 70, 75, 80, 100]

# SHAP 인사이트 패널용 — 파생변수 표에서 쓰는 한글 이름과 맞춤(표와 나란히 읽히도록).
SHAP_FEATURE_LABELS = {
    "pH_std": "pH 표준편차", "Temp_std": "온도 표준편차", "Voltage_std": "전압 표준편차",
    "pH_min": "pH 최솟값", "Temp_min": "온도 최솟값", "Voltage_min": "전압 최솟값",
    "pH_IQR": "pH IQR", "Temp_IQR": "온도 IQR", "Voltage_IQR": "전압 IQR",
    "pH_exc_high_rate": "pH 이탈률", "Temp_exc_high_rate": "온도 이탈률",
    "Voltage_exc_high_rate": "전압 이탈률",
}


def load_lot_summary() -> pd.DataFrame:
    df = pd.read_csv(BASE / "dashboard_lot_summary.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


@st.cache_data
def get_defect_lean_reference() -> dict:
    """파생변수 표 색칠용 — 726 LOT 전체(팀 핸드오프 dashboard_lot_summary.csv)에서
    정상(Defect=0)/불량(Defect=1) 그룹별 중앙값을 계산해, "이 값이 정상 쪽에 가까운지
    불량 쪽에 가까운지"를 판단할 기준점으로 쓴다(요청 반영, 2026-09 — 기존엔 현재
    표에 보이는 3개 행(pH/온도/전압)끼리만 비교하는 상대 히트맵이었는데, 정상/불량
    이라는 절대적 기준이 있는 두 점 사이의 위치로 바꿔달라는 요청).
    반환: {"pH_std": (정상 중앙값, 불량 중앙값), ...} — 12개 피처 전부."""
    lot_summary = load_lot_summary()
    features = [
        "pH_std", "Temp_std", "Voltage_std",
        "pH_min", "Temp_min", "Voltage_min",
        "pH_IQR", "Temp_IQR", "Voltage_IQR",
    ]
    med = lot_summary.groupby("Defect")[features].median()
    return {feat: (float(med.loc[0, feat]), float(med.loc[1, feat])) for feat in features}


def load_model_validation() -> pd.DataFrame:
    """팀원 핸드오프의 모델 검증 결과(PR-AUC/Recall/FPR/채택여부). 지금까지 대시보드에서
    한 번도 안 쓰이던 파일이었는데, "모델 정보" 섹션(사용 모델+핵심 지표)에 그대로
    쓰기 위해 추가함(2026-09, 튜터 피드백 반영한 관리자 탭 설계)."""
    return pd.read_csv(BASE / "dashboard_model_validation.csv")


def load_progress_checkpoint() -> pd.DataFrame:
    df = pd.read_csv(BASE / "dashboard_progress_checkpoint.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def load_daily_summary() -> pd.DataFrame:
    df = pd.read_csv(BASE / "dashboard_daily_summary.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def load_reference() -> pd.DataFrame:
    return pd.read_csv(BASE / "dashboard_reference.csv")


def get_final_oof(lot_summary: pd.DataFrame, date, slot: int) -> dict:
    """완료 LOT의 OOF 최종 판정 (과거 화면용, 누수 없음)."""
    row = lot_summary[(lot_summary["Date"] == date) & (lot_summary["Lot"] == slot)]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "prediction": row["Final_OOF_Prediction"],
        "risk_percentile": row["Final_OOF_Risk_Percentile"],
        "actual_label": row["Actual_Label"],
    }


def get_ew_oof(progress_cp: pd.DataFrame, date, slot: int, progress: int) -> dict:
    """특정 진행률 체크포인트의 OOF 조기경보 참고값 (과거 화면용)."""
    row = progress_cp[
        (progress_cp["Date"] == date) & (progress_cp["Lot"] == slot) & (progress_cp["Progress"] == progress)
    ]
    if row.empty:
        return {}
    row = row.iloc[0]
    return {
        "risk_percentile": row["EW_OOF_Risk_Percentile"],
        "threshold_alert": bool(row["EW_OOF_Threshold_Alert"]),
        "prefix_length": int(row["Prefix_Length"]),
        "pH_current": row["pH_Current"],
        "Temp_current": row["Temp_Current"],
        "Voltage_current": row["Voltage_Current"],
        "pH_std": row["pH_std"], "pH_min": row["pH_min"], "pH_IQR": row["pH_IQR"],
        "pH_display_excursion_rate": row["pH_Display_Excursion_Rate"],
        "Temp_std": row["Temp_std"], "Temp_min": row["Temp_min"], "Temp_IQR": row["Temp_IQR"],
        "Temp_display_excursion_rate": row["Temp_Display_Excursion_Rate"],
        "Voltage_std": row["Voltage_std"], "Voltage_min": row["Voltage_min"], "Voltage_IQR": row["Voltage_IQR"],
        "Voltage_display_excursion_rate": row["Voltage_Display_Excursion_Rate"],
        "actual_label": row["Actual_Label"],
    }


def load_timeseries() -> pd.DataFrame:
    """team 핸드오프의 시계열 CSV — Z-score, Progress_pct 등 이미 계산되어 있음."""
    df = pd.read_csv(BASE / "dashboard_sensor_timeseries.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def get_lot_timeseries(ts: pd.DataFrame, date, slot: int) -> pd.DataFrame:
    sub = ts[(ts["Date"] == date) & (ts["Lot"] == slot)]
    return sub.sort_values("Measurement_No").reset_index(drop=True)


def get_golden_batch_profile(ts: pd.DataFrame) -> pd.DataFrame:
    """"Golden Batch" 기준 궤적 — 정상(Defect==0) LOT들의 측정 순번(Measurement_No)별
    pH/온도/전압 Z-score 평균. 모든 LOT이 정확히 69개 측정치로 길이가 동일해서
    (실측 확인함), 진행률(%) 정렬 같은 보정 없이 순번별로 바로 평균 낼 수 있다.
    combined_zscore_chart에 "정상 LOT 평균" 참고선으로 얹어서, 선택한 LOT이 정상
    궤적 대비 언제부터·얼마나 벗어났는지 한눈에 비교하는 용도(팀원이 참고한 타사
    대시보드의 "Golden Batch 프로파일 비교"를 우리 데이터로 구현한 것)."""
    normal = ts[ts["Defect"] == 0]
    profile = normal.groupby("Measurement_No")[
        ["pH_Z_Display", "Temp_Z_Display", "Voltage_Z_Display"]
    ].mean().reset_index()
    return profile


def get_defect_batch_range(ts: pd.DataFrame) -> pd.DataFrame:
    """불량(Defect==1) LOT 9개의 측정 순번별 Z-score 25~75백분위수(IQR) 범위.
    Golden Batch 평균선 옆에 음영 밴드로 겹쳐서 "정상은 이렇게 모여있고 불량은
    이렇게 벗어난다"를 분포로 보여주는 용도(관리자 탭 브레인스토밍 반영).
    처음엔 min/max를 썼는데, 9개뿐이라 값 변동이 큰 지점에서 밴드 폭이 순간적으로
    5 이상 튀어서 그래프를 뒤덮어버리는 문제가 있었음(실측 확인) — 극단적인 LOT
    1~2개에 끌려다니지 않게 25~75백분위수(중간 절반)로 바꿔서 폭을 훨씬 안정적으로
    (평균 약 1.4 수준) 만듦. 극단 사례는 어차피 LOT을 직접 선택해서 볼 수 있다."""
    defect = ts[ts["Defect"] == 1]
    cols = ["pH_Z_Display", "Temp_Z_Display", "Voltage_Z_Display"]
    rng = defect.groupby("Measurement_No")[cols].quantile([0.25, 0.75]).unstack()
    rng.columns = [f"{col}_{('q25' if q == 0.25 else 'q75')}" for col, q in rng.columns]
    return rng.reset_index()


def get_golden_batch_profile_raw(ts: pd.DataFrame) -> pd.DataFrame:
    """get_golden_batch_profile()의 원본 단위(raw) 버전 — 정상 LOT들의 측정 순번별
    pH/온도/전압 실제값(단위 그대로) 평균. 관리도(variable_control_chart)는 Z-score가
    아니라 원본 단위 축을 쓰기 때문에, 거기에 겹쳐 그리려면 이 원본 단위 평균이 필요함
    (요청 반영, 2026-09 — "관리도에도 Golden Batch 방식으로 표기")."""
    normal = ts[ts["Defect"] == 0]
    profile = normal.groupby("Measurement_No")[["pH", "Temp", "Voltage"]].mean().reset_index()
    return profile


def get_defect_batch_range_raw(ts: pd.DataFrame) -> pd.DataFrame:
    """get_defect_batch_range()의 원본 단위(raw) 버전 — 불량 LOT 9개의 측정 순번별
    pH/온도/전압 실제값 25~75백분위수(IQR) 범위. 관리도용."""
    defect = ts[ts["Defect"] == 1]
    cols = ["pH", "Temp", "Voltage"]
    rng = defect.groupby("Measurement_No")[cols].quantile([0.25, 0.75]).unstack()
    rng.columns = [f"{col}_{('q25' if q == 0.25 else 'q75')}" for col, q in rng.columns]
    return rng.reset_index()


def risk_status_label(pct: float) -> str:
    """risk_percentile(0~1) -> 낮음/주의/높음 라벨 (README 기준과 동일)."""
    if pct >= 0.97:
        return "높음"
    if pct >= 0.90:
        return "주의"
    return "낮음"


def get_out_of_control_rate(ts: pd.DataFrame, start_d, end_d) -> dict:
    """선택 기간 내 pH/온도/전압 각각의 "관리이탈율"(넬슨룰 1·5번 위반 비율, %).
    Cpk를 검토했으나 진짜 외부 규격 한계가 없어(Display_Limit/Model_Ref 둘 다
    정상 데이터 자체에서 뽑은 값이라 순환적) 대신 채택한 지표 — 이미 그래프에서
    쓰는 넬슨룰 판정(chart_helpers._nelson_rule1/5)을 LOT별로 그대로 적용해 집계만
    한다(새 판정 로직 아님, 계산 일관성 보장). Rule 5는 연속 3점 슬라이딩 윈도우라
    LOT 경계를 넘으면 안 되므로 LOT 단위로 나눠서 계산한다."""
    from chart_helpers import _nelson_rule1, _nelson_rule5  # 순환참조 없음(chart_helpers는 pandas/altair만 의존)

    period = ts[(ts["Date"] >= start_d) & (ts["Date"] <= end_d)]
    col_map = {"pH": "pH_Z_Display", "온도": "Temp_Z_Display", "전압": "Voltage_Z_Display"}
    rates = {}
    for label, col in col_map.items():
        total = 0
        violations = 0
        for _, g in period.groupby(["Date", "Lot"]):
            z = g.sort_values("Measurement_No")[col]
            viol = _nelson_rule1(z) | _nelson_rule5(z)
            violations += int(viol.sum())
            total += len(z)
        rates[label] = round(100 * violations / total, 1) if total else None
    return rates


def get_period_nelson_violations(ts: pd.DataFrame, start_d, end_d) -> dict:
    """선택 기간 전체의 pH/온도/전압 넬슨룰 위반 "건수"까지 포함한 버전
    — get_out_of_control_rate()는 비율(%)만 반환하는데, AI 자동 리포트의
    "관리도 요약" 섹션에서 선택 LOT 건수(get_lot_nelson_violations)와 나란히
    비교하려면 같은 형식(rule1_count/rule5_count/total_points)이 필요해서
    추가함(2026-09). 판정 로직은 get_out_of_control_rate와 동일(새 로직 아님).
    반환: {"pH": {"rule1_count":.., "rule5_count":.., "total_points":.., "rate":..}, ...}"""
    from chart_helpers import _nelson_rule1, _nelson_rule5

    period = ts[(ts["Date"] >= start_d) & (ts["Date"] <= end_d)]
    col_map = {"pH": "pH_Z_Display", "온도": "Temp_Z_Display", "전압": "Voltage_Z_Display"}
    out = {}
    for label, col in col_map.items():
        total = 0
        rule1_count = 0
        rule5_count = 0
        for _, g in period.groupby(["Date", "Lot"]):
            z = g.sort_values("Measurement_No")[col]
            rule1 = _nelson_rule1(z)
            rule5 = _nelson_rule5(z)
            rule1_count += int(rule1.sum())
            rule5_count += int((rule5 & ~rule1).sum())
            total += len(z)
        rate = round(100 * (rule1_count + rule5_count) / total, 1) if total else None
        out[label] = {
            "rule1_count": rule1_count, "rule5_count": rule5_count,
            "total_points": total, "rate": rate,
        }
    return out


def get_lot_nelson_violations(lot_ts: pd.DataFrame) -> dict:
    """선택 LOT 하나의 관리도 기준 넬슨룰 위반 건수(Rule 1=±3σ 초과, Rule 5=조기경고)
    — chart_helpers의 관리도·get_out_of_control_rate와 동일한 판정 로직
    (_nelson_rule1/5)을 그대로 재사용해 개수만 집계한다(새 판정 로직 아님, 계산
    일관성 보장). AI 자동 리포트의 "관리도 요약" 섹션 입력으로 씀(2026-09).
    반환: {"pH": {"rule1_count":.., "rule5_count":.., "total_points":..}, ...}"""
    from chart_helpers import _nelson_rule1, _nelson_rule5

    col_map = {"pH": "pH_Z_Display", "온도": "Temp_Z_Display", "전압": "Voltage_Z_Display"}
    out = {}
    for label, col in col_map.items():
        z = lot_ts[col]
        rule1 = _nelson_rule1(z)
        rule5 = _nelson_rule5(z)
        out[label] = {
            "rule1_count": int(rule1.sum()),
            "rule5_count": int((rule5 & ~rule1).sum()),
            "total_points": int(len(z)),
        }
    return out


def get_recent_defect_rate_trend(df: pd.DataFrame, end_d) -> dict:
    """선택한 조회 기간의 마지막 7일 vs 그 직전 7일의 불량률(%) 비교.
    처음엔 데이터 전체의 최신 날짜를 기준으로 고정 계산했는데, 사용자가 사이드바에서
    고른 조회 기간과 맞물리게 해달라는 요청으로 변경(2026-09) — 기준일(end_d)을
    "선택한 기간의 끝 날짜"로 받아서 그 날짜 기준으로 최근 7일/그 이전 7일을 계산한다.
    비교 대상 7일은 선택 기간 밖으로 걸쳐도 되므로(예: 기간이 3일만 선택된 경우도
    비교가 가능해야 함), 항상 전체 df에서 값을 가져온다 — period_df가 아니라 df를
    받는 이유. 데이터가 14일 미만이면 비교 불가로 None 반환."""
    max_date = end_d
    recent_start = max_date - pd.Timedelta(days=6)
    prev_end = recent_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=6)
    if prev_start < df["date"].min():
        return {"현재": None, "이전": None, "변화": None}

    def _defect_rate(sub: pd.DataFrame):
        lots = sub.groupby(["date", "lot"])["FinalError"].max()
        return 100 * lots.mean() if len(lots) else None

    recent = df[(df["date"] >= recent_start) & (df["date"] <= max_date)]
    prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]
    cur_rate = _defect_rate(recent)
    prev_rate = _defect_rate(prev)
    delta = round(cur_rate - prev_rate, 1) if cur_rate is not None and prev_rate is not None else None
    return {"현재": cur_rate, "이전": prev_rate, "변화": delta}


@st.cache_resource
def _load_shap_artifacts():
    """SHAP 이탈진단용 실제 모델 아티팩트(채택 모델: XGBoost 12F).
    chromate_dashboard_handoff/models 안의 원본 팀원 핸드오프 산출물을 그대로 씀 —
    "모델 정보" 카드가 참조하는 dashboard_model_validation.csv와 같은 모델이다.
    st.cache_resource로 세션당 1회만 로드(모델 로드+TreeExplainer 생성 비용 있음)."""
    fc = json.loads((SHAP_MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
    fp = joblib.load(SHAP_MODEL_DIR / "final_preprocess.joblib")
    fm = XGBClassifier()
    fm.load_model(str(SHAP_MODEL_DIR / "final_defect_xgb.json"))
    explainer = shap.TreeExplainer(fm)
    return fc, fp, explainer


def get_shap_contributions(lot_df: pd.DataFrame) -> pd.DataFrame:
    """선택한 LOT 하나에 대해, 채택 모델(XGBoost 12F)이 실제로 어느 피처를 근거로
    판정했는지 SHAP 기여도로 계산한다. 인사이트 섹션의 규칙기반(이탈률 기준) 설명과는
    별개 패널로 병행 표시하기 위한 것(대체 아님, 2026-09 요청 반영).

    피처 계산식은 팀 핸드오프의 chromate_dashboard_handoff/code/dashboard_inference.py
    predict_final_lot()과 동일하게 맞춤(모델 자체의 excursion_reference 기준 사용 —
    화면 표시용 파생변수 표의 기준(2.20/40/15)과는 다를 수 있음, 학습 당시 기준 그대로).

    반환: feature(원 이름)/label(한글)/value(피처 값)/shap(기여도, 값이 클수록
    "불량" 쪽으로, 작을수록(음수) "정상" 쪽으로 민 정도) — |shap| 내림차순.
    """
    fc, fp, explainer = _load_shap_artifacts()
    x = lot_df.rename(columns={"ph": "pH", "temp": "Temp", "voltage": "Voltage"})
    ref = fc["excursion_reference"]
    feats = {}
    for s in ["pH", "Temp", "Voltage"]:
        v = x[s].to_numpy(float)
        feats[f"{s}_std"] = float(np.std(v, ddof=1))
        feats[f"{s}_min"] = float(v.min())
        feats[f"{s}_IQR"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        feats[f"{s}_exc_high_rate"] = float(np.mean(v > ref[s]["upper"]))

    z = pd.DataFrame([feats])[fc["features"]].fillna(fp["median"])
    z_scaled = fp["scaler"].transform(z.to_numpy(np.float32))
    shap_vals = np.asarray(explainer.shap_values(z_scaled)).reshape(-1)

    out = pd.DataFrame({
        "feature": fc["features"],
        "label": [SHAP_FEATURE_LABELS.get(f, f) for f in fc["features"]],
        "value": [feats[f] for f in fc["features"]],
        "shap": shap_vals,
    })
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)
