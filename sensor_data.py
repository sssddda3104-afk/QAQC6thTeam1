"""
정형(센서) 데이터 처리 함수 모음.
- CSV 로딩
- LOT(날짜×슬롯) 단위 목록/요약
- 파생변수(표준편차·최솟값·IQR·이탈률) 계산

주의: `lot` 컬럼(1~22)은 실제 LOT 번호가 아니라 슬롯 번호다.
실제 LOT은 "날짜 + 슬롯" 조합이며, 같은 날짜 안에서는 슬롯들이
동시가 아니라 순차적으로 처리된다 (대화에서 실측 확인함).

이탈 기준(팀 정의):
- pH:  상단(2.20 초과)
- 온도: 하단(40 미만)
- 전압: 하단(15 미만)
"""

from pathlib import Path
import pandas as pd
import numpy as np

CSV_PATH = Path(__file__).parent / "data" / "merged_sensor_data_with_error.csv"

# 변수별 이탈 판정 기준 (컬럼명, 기준값, 방향)
VAR_BOUNDS = {
    "pH": ("ph", 2.20, "upper"),
    "온도": ("temp", 40, "lower"),
    "전압": ("voltage", 15, "lower"),
}


def get_normal_reference_stats(df: pd.DataFrame) -> dict:
    """정상(FinalError==0) LOT 전체 데이터 기준 pH/온도/전압 평균·표준편차.

    Nelson Rule(관리도) 판정의 기준선(중심선)·시그마 밴드 계산에 쓴다.
    반드시 정상 데이터에만 fit — 팀이 기존에 8-Fold OOF·정상-only Calibration에서
    쓴 것과 동일한 leakage 방지 원칙(Reference는 Development에만 fit)을 따른다."""
    normal = df[df["FinalError"] == 0]
    stats = {}
    for label, (col, _, _) in VAR_BOUNDS.items():
        stats[label] = {"mean": float(normal[col].mean()), "std": float(normal[col].std())}
    return stats


def load_raw() -> pd.DataFrame:
    """CSV를 읽어 datetime/date 컬럼을 붙여 반환한다."""
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    return df


def get_lot_list(df: pd.DataFrame) -> pd.DataFrame:
    """(date, lot) 조합별 1행 요약 — 전체 LOT 목록과 양/불 라벨.

    사용자에게는 날짜 없이 "LOT-0001" 같은 단일 번호만 노출하기로 했으므로
    시간순 정렬 후 순번을 매긴 lot_id 컬럼을 같이 붙인다.
    """
    g = df.groupby(["date", "lot"])
    summary = g.agg(
        error=("FinalError", "max"),
        n=("ph", "size"),
    ).reset_index()
    summary = summary.sort_values(["date", "lot"]).reset_index(drop=True)
    summary["lot_id"] = [f"LOT-{i + 1:04d}" for i in range(len(summary))]
    return summary


def get_lot_by_id(lot_list: pd.DataFrame, lot_id: str):
    """lot_id로 (date, slot) 튜플을 찾는다."""
    row = lot_list.loc[lot_list["lot_id"] == lot_id].iloc[0]
    return row["date"], int(row["lot"])


def get_overview_kpi(df: pd.DataFrame) -> dict:
    """1페이지 상단 KPI(현장형 집계값)에 쓰는 값들.

    선택한 기간에 데이터가 하나도 없으면(0건) 나눗셈 등이 터지지 않도록
    해당 항목들은 None으로 반환한다 — 화면에서는 None을 "-"로 표시한다.
    """
    if df.empty:
        return {
            "조회 기간(일)": 0,
            "총 LOT 수": 0,
            "불량 LOT 수": 0,
            "양품률(%)": None,
            "불량률(%)": None,
            "평균 pH": None,
            "평균 온도": None,
            "평균 전압": None,
            "가동 슬롯 수": 0,
        }

    lots = get_lot_list(df)
    total_lots = len(lots)
    bad_lots = int(lots["error"].sum())
    good_rate = 100 * (1 - bad_lots / total_lots) if total_lots > 0 else None
    bad_rate = round(100 * bad_lots / total_lots, 1) if total_lots > 0 else None
    return {
        "조회 기간(일)": df["date"].nunique(),
        "총 LOT 수": total_lots,
        "불량 LOT 수": bad_lots,
        "양품률(%)": round(good_rate, 1) if good_rate is not None else None,
        "불량률(%)": bad_rate,
        "평균 pH": round(df["ph"].mean(), 2),
        "평균 온도": round(df["temp"].mean(), 1),
        "평균 전압": round(df["voltage"].mean(), 1),
        "가동 슬롯 수": df["lot"].nunique(),
    }


def get_lot_subset(df: pd.DataFrame, date, slot: int) -> pd.DataFrame:
    """선택한 (날짜, 슬롯)에 해당하는 원본 시계열 행들 (시간순 정렬)."""
    sub = df[(df["date"] == date) & (df["lot"] == slot)].sort_values("datetime")
    return sub.reset_index(drop=True)


def derived_features(series: pd.Series, bound: float, direction: str) -> dict:
    """표준편차·최솟값·IQR·이탈률 하나를 계산."""
    std = float(series.std())
    mn = float(series.min())
    q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = float(q3 - q1)
    if direction == "upper":
        rate = float((series > bound).mean() * 100)
    else:
        rate = float((series < bound).mean() * 100)
    return {"표준편차": round(std, 3), "최솟값": round(mn, 2), "IQR": round(iqr, 3), "이탈률(%)": round(rate, 1)}


def get_variable_table_from_ts(lot_ts, upto: int | None = None):
    """
    팀 핸드오프의 timeseries(컬럼명 pH/Temp/Voltage, 대문자)용 파생변수 표.
    get_variable_table과 로직은 같고 컬럼명 매핑만 다르다.
    """
    bounds_ts = {
        "pH": ("pH", 2.20, "upper"),
        "온도": ("Temp", 40, "lower"),
        "전압": ("Voltage", 15, "lower"),
    }
    rows = []
    sliced = lot_ts if upto is None else lot_ts.iloc[:upto]
    for name, (col, bound, direction) in bounds_ts.items():
        feats = derived_features(sliced[col], bound, direction)
        feats["변수"] = name
        feats["방향"] = f"{'상단' if direction == 'upper' else '하단'} {bound} 기준"
        feats["현재값"] = round(float(sliced[col].iloc[-1]), 2)
        rows.append(feats)
    cols = ["변수", "현재값", "표준편차", "최솟값", "IQR", "이탈률(%)", "방향"]
    return pd.DataFrame(rows)[cols]


def get_variable_table(lot_df: pd.DataFrame, upto: int | None = None) -> pd.DataFrame:
    """
    파생변수 표 (pH/온도/전압 x 표준편차/최솟값/IQR/이탈률).
    upto가 주어지면 그 지점까지만 잘라서 계산 (실시간뷰 진행중 시뮬레이션용).
    """
    rows = []
    sliced = lot_df if upto is None else lot_df.iloc[:upto]
    for name, (col, bound, direction) in VAR_BOUNDS.items():
        feats = derived_features(sliced[col], bound, direction)
        feats["변수"] = name
        feats["방향"] = f"{'상단' if direction == 'upper' else '하단'} {bound} 기준"
        if upto is not None:
            feats["현재값"] = round(float(sliced[col].iloc[-1]), 2)
        rows.append(feats)
    cols = ["변수"] + (["현재값"] if upto is not None else []) + ["표준편차", "최솟값", "IQR", "이탈률(%)", "방향"]
    return pd.DataFrame(rows)[cols]
