"""
정형 데이터 페이지의 그래프(통합 Z-score 추이 + 변수별 관리도) 구성.
Altair 사용 (Streamlit 내장 의존성이라 별도 설치 불필요).
"""
import altair as alt
import pandas as pd

# 변수별 표시 기준(팀 정의) — (컬럼명, 기준값, 방향)
DISPLAY_BOUNDS = {
    "pH": ("pH", 2.20, "upper"),
    "온도": ("Temp", 40, "lower"),
    "전압": ("Voltage", 15, "lower"),
}


def defect_heatmap_chart(lots: pd.DataFrame) -> alt.Chart:
    """전체 LOT을 날짜(x축)×LOT 슬롯 번호(y축) 정사각형 격자로 표시. 불량 LOT만 강조색.
    바코드(1차원 나열)보다 "어느 날짜의 몇 번째 슬롯에서 불량이 났는지" 2차원으로
    한눈에 보여달라는 팀 피드백 반영(2026-09). 날짜마다 슬롯 수가 정확히 22개로
    똑같아서(실측 확인, 33일×22슬롯=726) 빈 칸 없는 완전한 격자가 나온다.
    정상/불량을 별도 레이어로 나눠서, 정상 칸은 tooltip=False로 마우스를 올려도
    아무 반응이 없게 하고(팀 피드백), 불량 칸에만 호버 시 LOT 정보가 뜬다.
    width/height를 alt.Step(고정 픽셀)으로 줘서 칸이 직사각형이 아니라 정사각형이
    되게 함(가로 33칸·세로 22칸이라 step을 안 고정하면 셀이 옆으로 퍼져 보임).

    (칸 클릭 시 날짜/LOT 선택을 자동으로 이동시키는 기능을 시도했었으나, 실제
    브라우저에서 클릭이 안정적으로 등록되지 않는 문제를 끝내 해결 못해 보류함(2026-09)
    — 대신 app.py의 옆 패널에 "불량 발생 목록" + 조회 버튼(st.button, 훨씬 안정적인
    방식)으로 같은 요구를 충족시킴.)"""
    data = lots.copy()
    data["날짜"] = data["date"].astype(str)
    data["상태"] = data["error"].map({0: "정상", 1: "불량"})
    data["lot_label"] = data["date"].astype(str) + " · Lot " + data["lot"].astype(str)

    x_enc = alt.X("날짜:O", title="날짜", axis=alt.Axis(labelAngle=-45))
    y_enc = alt.Y("lot:O", title="LOT 슬롯 번호", sort="descending")

    normal = (
        alt.Chart(data)
        .transform_filter(alt.datum.상태 == "정상")
        .mark_rect(color="#e5e9e7", stroke="#ffffff", strokeWidth=1, tooltip=False)
        .encode(x=x_enc, y=y_enc)
    )
    defect = (
        alt.Chart(data)
        .transform_filter(alt.datum.상태 == "불량")
        .mark_rect(color="#dc2626", stroke="#ffffff", strokeWidth=1)
        .encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip("lot_label:N", title="LOT"),
                alt.Tooltip("상태:N", title="결과"),
            ],
        )
    )
    return (normal + defect).properties(width=alt.Step(14), height=alt.Step(14))


def defect_barcode_chart(lots: pd.DataFrame) -> alt.Chart:
    """전체 LOT을 날짜순으로 얇은 눈금 하나씩(총 726개) 나열해서, 불량 LOT만
    강조색으로 표시하는 '바코드' 차트. 파이 차트처럼 비율(1.2%)이 왜곡돼 보이지
    않으면서, 불량이 특정 시기에 몰려 있는지도 같이 보여준다.
    x축은 LOT 단위(순번)로 촘촘하게 유지하되(예: 2021-09-09처럼 같은 날 불량이
    2건 겹치는 경우도 놓치지 않도록), 축에 표시되는 라벨만 날짜 단위로 보여준다.
    정상/불량을 서로 다른 레이어로 그려서, 마우스를 올렸을 때 불량 LOT에만
    툴팁이 뜨고 정상 LOT은 반응하지 않게 한다.
    (2026-09: 팀 피드백으로 날짜×슬롯 히트맵(defect_heatmap_chart)으로 교체됨 —
    이 함수는 화면에서 안 쓰이지만 참고용으로 남겨둠.)"""
    data = pd.DataFrame({
        "순번": range(1, len(lots) + 1),
        "날짜": lots["date"].astype(str).values,
        "상태": lots["error"].map({0: "정상", 1: "불량"}).values,
        "lot_label": (lots["date"].astype(str) + " · Lot " + lots["lot"].astype(str)).values,
    })

    # 날짜가 바뀌는 첫 순번들만 골라서 그 위치에만 날짜 라벨을 붙인다
    # (매 LOT마다 라벨을 달면 726개가 다 겹쳐 보이므로).
    first_idx_per_date = (
        data.groupby("날짜")["순번"].min().sort_values().astype(int).tolist()
    )
    date_labels = data.groupby("날짜")["순번"].min().sort_values().index.tolist()
    js_dates = "[" + ",".join(f"'{d}'" for d in date_labels) + "]"
    js_values = "[" + ",".join(str(v) for v in first_idx_per_date) + "]"

    x_enc = alt.X(
        "순번:Q",
        title=None,
        axis=alt.Axis(
            values=first_idx_per_date,
            labelExpr=f"{js_dates}[indexof({js_values}, datum.value)]",
            labelAngle=-45,
            grid=False,
        ),
    )

    normal = (
        alt.Chart(data)
        .transform_filter(alt.datum.상태 == "정상")
        .mark_tick(thickness=2, size=40, color="#5b9bd5", tooltip=False)
        .encode(x=x_enc)
    )
    defect = (
        alt.Chart(data)
        .transform_filter(alt.datum.상태 == "불량")
        .mark_tick(thickness=2, size=40, color="#ff4b4b")
        .encode(
            x=x_enc,
            tooltip=[
                alt.Tooltip("lot_label:N", title="LOT"),
                alt.Tooltip("상태:N", title="결과"),
            ],
        )
    )
    return (normal + defect).properties(height=110)


def combined_zscore_chart(
    lot_ts: pd.DataFrame,
    show_vars: dict | None = None,
    golden_batch: pd.DataFrame | None = None,
    defect_range: pd.DataFrame | None = None,
) -> alt.Chart:
    """3대 변수 Z-score 통합 추이 (팀이 계산한 *_Z_Display 컬럼 사용).

    x축은 측정 시각(Timestamp) 기준. 마우스를 올리면 가장 가까운 시점에
    세로선 + 강조점 + 툴팁이 뜨고, 범례를 클릭하면 그 변수만 강조된다.
    점을 클릭하면 그 시점의 lot_ts 행 번호(정수 "_idx")가 "point_select" 파라미터로
    넘어가므로, 호출부에서 st.altair_chart(..., on_select="rerun",
    selection_mode=["point_select"])로 받아 lot_ts.iloc[idx]로 그 시점 값을 바로
    조회해 화면 옆에 표시할 수 있다. (처음엔 "시각"(타임스탬프) 필드로 셀렉션을
    잡았었는데, 정수 대신 타임스탬프를 프론트엔드-파이썬 간에 왕복시키는 과정에서
    한 사용자 환경에서 클릭해도 항상 같은 값만 나오는 문제가 있었음 — 정수 인덱스는
    타임존/포맷 변환 여지가 전혀 없어 더 안전하다고 판단해 이 방식으로 교체함. 2026-09.)
    show_vars: {"pH": bool, "온도": bool, "전압": bool} — False인 변수는 선에서 제외.
    golden_batch: model_data.get_golden_batch_profile()의 결과(Measurement_No별 정상
    LOT 평균 Z-score). 주어지면 lot_ts의 Measurement_No에 맞춰 같은 x축(시각) 위에
    점선으로 겹쳐 그린다("Golden Batch" 비교선) — 관리자 탭 전용, 없으면 생략.
    defect_range: model_data.get_defect_batch_range()의 결과(Measurement_No별 불량
    LOT 9개의 min~max Z-score). 주어지면 변수별로 옅은 음영 밴드로 겹쳐서 "불량
    LOT들이 어느 범위에 분포하는지" 보여준다 — 관리자 탭 전용, 없으면 생략.
    """
    show_vars = show_vars or {"pH": True, "온도": True, "전압": True}

    def _seq(ts):
        return ts["Measurement_No"].astype(str) + " / " + ts["Total_Points"].astype(str)

    idx = list(range(len(lot_ts)))

    parts = []
    if show_vars.get("pH", True):
        parts.append(pd.DataFrame({
            "시각": lot_ts["Timestamp"], "Z-score": lot_ts["pH_Z_Display"], "변수": "pH",
            "측정순번": _seq(lot_ts), "_idx": idx,
        }))
    if show_vars.get("온도", True):
        parts.append(pd.DataFrame({
            "시각": lot_ts["Timestamp"], "Z-score": lot_ts["Temp_Z_Display"], "변수": "온도",
            "측정순번": _seq(lot_ts), "_idx": idx,
        }))
    if show_vars.get("전압", True):
        parts.append(pd.DataFrame({
            "시각": lot_ts["Timestamp"], "Z-score": lot_ts["Voltage_Z_Display"], "변수": "전압",
            "측정순번": _seq(lot_ts), "_idx": idx,
        }))
    if not parts:
        long_df = pd.DataFrame({"시각": [], "Z-score": [], "변수": [], "측정순번": [], "_idx": []})
    else:
        long_df = pd.concat(parts)

    # 범례 클릭으로만 변수 강조를 전환하도록 bind="legend"로 명시적으로 스코프를 좁힘
    # (예전엔 on="click"이라 차트 아무 곳이나 클릭해도 반응해서, 다른 상호작용과 겹치는
    # 문제가 있었음).
    # ⚠ clear="dblclick"를 bind="legend"와 같이 쓰면 실제 브라우저에서 Vega 내부 오류
    # ("t.events is not iterable")가 나면서 그래프 자체가 아예 안 그려지는 문제가 있었음
    # (Python/Altair 스펙 생성 자체는 에러 없이 통과해서 이 저장소의 서버 쪽 테스트로는
    # 못 잡았고, 실제 브라우저에서 렌더링해봐야 드러남) — bind="legend"는 범례 클릭 시
    # 자체적으로 토글되므로 별도 clear가 필요 없어 제거해서 해결.
    click_sel = alt.selection_point(fields=["변수"], bind="legend", empty="all")
    hover = alt.selection_point(fields=["시각"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    # 점 클릭 시 그 시점(lot_ts 행 번호)을 Streamlit(파이썬) 쪽으로 돌려주기 위한 선택
    # 파라미터. name을 고정해야 앱 쪽에서
    # st.altair_chart(..., selection_mode=["point_select"])로 정확히 이 파라미터를
    # 지정해 값을 읽어올 수 있다. "nearest=True"라 x축(시각) 상 가장 가까운 점을 잡되,
    # 되돌려주는 값은 타임스탬프가 아니라 정수 "_idx"라서 왕복 과정에 파싱 오류 여지가 없다.
    point_select = alt.selection_point(
        fields=["_idx"], nearest=True, on="click", empty=False, name="point_select"
    )

    base = alt.Chart(long_df).encode(
        x=alt.X("시각:T", title="측정 시각", axis=alt.Axis(format="%H:%M:%S", tickCount=12)),
        y=alt.Y("Z-score:Q", title="표준화 값 (Z-score)"),
        color=alt.Color("변수:N", legend=alt.Legend(title=None, orient="top")),
    )

    line = base.mark_line().encode(
        opacity=alt.condition(click_sel, alt.value(1), alt.value(0.2)),
    ).add_params(click_sel)

    # 클릭으로 강조된 변수만 남기고, 그 안에서 마우스에 가장 가까운 시점 찾기
    # 점(point) 레이어들이 같은 color 인코딩을 공유하면 범례가 동그라미(선택버튼처럼) 바뀌므로
    # 점 레이어의 범례 기여는 꺼서, 선(line) 레이어의 범례만 남긴다 — 순수 표시용 범례.
    no_legend_color = alt.Color("변수:N", legend=None)

    selectors = base.mark_point(size=200).encode(
        color=no_legend_color,
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip("시각:T", title="시각", format="%H:%M:%S"),
            alt.Tooltip("측정순번:N", title="측정 순번(전체 중)"),
            alt.Tooltip("변수:N", title="변수"),
            alt.Tooltip("Z-score:Q", title="Z-score", format=".2f"),
        ],
    ).transform_filter(click_sel).add_params(hover, point_select)

    points = base.mark_point(size=70).encode(
        color=no_legend_color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    ).transform_filter(click_sel)

    rule = base.mark_rule(color="gray", strokeDash=[3, 3]).encode(
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
    ).transform_filter(hover)

    layers = [line, selectors, points, rule]

    if defect_range is not None and not defect_range.empty:
        # 불량 LOT 9개의 25~75백분위수(IQR) 범위를 변수별 음영 밴드로 겹쳐서 "불량은
        # 이런 범위에 분포한다"를 보여줌 — Golden Batch 평균선과 짝을 이루는 비교
        # 기준(관리자 탭). 맨 밑바닥(가장 먼저)에 깔아서 실측선·골든배치선을 안 가리게 한다.
        merged_d = lot_ts[["Measurement_No", "Timestamp"]].merge(
            defect_range, on="Measurement_No", how="left"
        )
        d_parts = []
        if show_vars.get("pH", True):
            d_parts.append(pd.DataFrame({
                "시각": merged_d["Timestamp"], "변수": "pH",
                "y1": merged_d["pH_Z_Display_q25"], "y2": merged_d["pH_Z_Display_q75"],
            }))
        if show_vars.get("온도", True):
            d_parts.append(pd.DataFrame({
                "시각": merged_d["Timestamp"], "변수": "온도",
                "y1": merged_d["Temp_Z_Display_q25"], "y2": merged_d["Temp_Z_Display_q75"],
            }))
        if show_vars.get("전압", True):
            d_parts.append(pd.DataFrame({
                "시각": merged_d["Timestamp"], "변수": "전압",
                "y1": merged_d["Voltage_Z_Display_q25"], "y2": merged_d["Voltage_Z_Display_q75"],
            }))
        if d_parts:
            defect_long = pd.concat(d_parts)
            defect_band = alt.Chart(defect_long).mark_area(opacity=0.15).encode(
                x="시각:T", y=alt.Y("y1:Q", title="표준화 값 (Z-score)"), y2="y2:Q",
                color=no_legend_color,
            ).transform_filter(click_sel)
            layers.insert(0, defect_band)

    if golden_batch is not None and not golden_batch.empty:
        # Measurement_No 기준으로 골든배치 평균을 이 LOT의 실제 시각에 맞춰 붙인다
        # (모든 LOT이 69개 측정치로 길이가 동일해서 순번 매칭만으로 충분 — 진행률
        # 정렬 같은 보정 불필요, 실측 확인함).
        merged = lot_ts[["Measurement_No", "Timestamp"]].merge(
            golden_batch, on="Measurement_No", how="left"
        )
        g_parts = []
        if show_vars.get("pH", True):
            g_parts.append(pd.DataFrame({
                "시각": merged["Timestamp"], "Z-score": merged["pH_Z_Display"], "변수": "pH",
            }))
        if show_vars.get("온도", True):
            g_parts.append(pd.DataFrame({
                "시각": merged["Timestamp"], "Z-score": merged["Temp_Z_Display"], "변수": "온도",
            }))
        if show_vars.get("전압", True):
            g_parts.append(pd.DataFrame({
                "시각": merged["Timestamp"], "Z-score": merged["Voltage_Z_Display"], "변수": "전압",
            }))
        if g_parts:
            golden_long = pd.concat(g_parts)
            golden_line = alt.Chart(golden_long).mark_line(
                strokeDash=[4, 3], strokeWidth=1.5, opacity=0.55
            ).encode(
                x="시각:T", y="Z-score:Q", color=no_legend_color,
            ).transform_filter(click_sel)
            # 실제 값 선(line)보다 먼저 그려서 뒤에 깔리게 함 — 비교 기준선이니 실측선을
            # 가리면 안 된다.
            layers.insert(0, golden_line)

    return alt.layer(*layers).properties(height=400)


def _nelson_rule1(z: pd.Series) -> pd.Series:
    """Rule 1: 1점이 중심선에서 ±3σ 밖."""
    return z.abs() > 3


def _nelson_rule5(z: pd.Series) -> pd.Series:
    """Rule 5: 연속 3점 중 2점이 같은 쪽 ±2σ 밖(조기 경고).
    슬라이딩 윈도우(현재 점 포함 최근 3점)로 판정.
    2σ 밴드 자체는 화면에 안 그리지만(팀 피드백), 판정 계산에는 그대로 2σ 기준을 쓴다."""
    n = len(z)
    violations = [False] * n
    z_vals = z.to_numpy()
    for i in range(n):
        window = z_vals[max(0, i - 2):i + 1]
        if len(window) < 3:
            continue
        if (window > 2).sum() >= 2 or (window < -2).sum() >= 2:
            violations[i] = True
    return pd.Series(violations, index=z.index)


def variable_control_chart(
    lot_ts: pd.DataFrame,
    var_name: str,
    ref_stats: dict | None = None,
    upto: int | None = None,
    show_sigma3: bool = False,
    show_iqr: bool = False,
    show_min: bool = True,
    golden_batch_raw: pd.DataFrame | None = None,
    defect_range_raw: pd.DataFrame | None = None,
) -> alt.Chart:
    """변수 하나의 관리도 — 실측값 + 넬슨룰(Rule 1·5) 위반 마커(항상 표시)
    + (선택) 3σ 밴드 + (선택) IQR 음영 + (선택) 최솟값 포인트.

    x축은 측정 시각(Timestamp) 기준이며, 마우스를 올리면 가장 가까운 시점에
    세로선 + 강조점 + 툴팁이 뜬다.
    ref_stats: {"mean": float, "std": float} — 정상 데이터 기준 평균·표준편차
    (sensor_data.get_normal_reference_stats()). None이면 넬슨룰 판정 자체를 건너뛴다.
    1σ 밴드는 안 보여준다(요청에 따라 제외) — 대신 중심선(평균)만 점선으로 표시.

    넬슨룰(Rule 1: ±3σ 밖 급성 이상, Rule 5: 3점 중 2점 ±2σ 밖 조기경고)은 팀 피드백에
    따라 옵션이 아니라 항상 자동 적용된다 — 위반 점에 마우스를 올리면 툴팁에 몇 번
    룰을 위반했는지(Rule 1 / Rule 5) 표시된다. 2σ 밴드는 화면에 그리지 않되(판정 계산에는
    여전히 사용), 3σ 밴드는 넬슨룰 자동판정과 별개로 show_sigma3 옵션으로 켜고 끌 수 있다.
    구 "팀 기준선"(DISPLAY_BOUNDS의 이탈 기준선, 예: pH>2.20) 옵션은 넬슨룰 자동판정과
    역할이 겹치고 팀 피드백으로 불필요하다고 판단되어 제거함 — DISPLAY_BOUNDS 자체는
    다른 곳(예: 파생변수 계산)에서 여전히 쓰이므로 남겨둔다.
    show_*: 각 오버레이를 켜고 끄는 체크박스용 플래그 — 전부 켜면 예전처럼 다 보이고,
    끄면 그 레이어만 빠진다.
    golden_batch_raw/defect_range_raw: model_data.get_golden_batch_profile_raw()/
    get_defect_batch_range_raw()의 결과(원본 단위) — combined_zscore_chart와 같은
    "Golden Batch" 비교 방식을 관리도에도 적용해달라는 요청 반영(2026-09). 정상 LOT
    평균을 점선으로, 불량 LOT 9개의 IQR 범위를 음영으로 겹쳐 그린다.
    """
    col = DISPLAY_BOUNDS[var_name][0]
    data = (lot_ts if upto is None else lot_ts.iloc[:upto]).copy()
    data["측정순번"] = data["Measurement_No"].astype(str) + " / " + data["Total_Points"].astype(str)
    # 넬슨룰 판정 결과를 전체 데이터에 미리 컬럼으로 심어둔다(기본값 "정상 범위") —
    # 아래 selectors(호버 감지용 큰 투명 히트영역) 레이어의 툴팁에도 이 컬럼을 넣어야
    # 위반 마커(rule1_pts/rule5_pts) 위에 마우스를 올렸을 때 실제로 판정 문구가 뜬다.
    # (레이어 순서상 selectors가 위반 마커보다 위에 그려져 마우스 이벤트를 먼저 가로채므로,
    # 위반 마커 레이어에만 툴팁을 넣으면 화면에 안 뜨는 문제가 있었음).
    data["넬슨룰 판정"] = "정상 범위"
    series = data[col]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    min_idx = series.idxmin()
    min_row = data.loc[[min_idx]]

    y_scale = alt.Scale(zero=False)
    data = data.rename(columns={"Timestamp": "시각"})
    min_row = min_row.rename(columns={"Timestamp": "시각"})

    hover = alt.selection_point(fields=["시각"], nearest=True, on="pointerover", empty=False, clear="pointerout")

    base = alt.Chart(data).encode(
        x=alt.X("시각:T", title="측정 시각", axis=alt.Axis(format="%H:%M:%S", tickCount=12)),
        y=alt.Y(f"{col}:Q", title=var_name, scale=y_scale),
    )

    layers = []

    if defect_range_raw is not None and not defect_range_raw.empty:
        merged_d = data[["Measurement_No", "시각"]].merge(defect_range_raw, on="Measurement_No", how="left")
        defect_band = alt.Chart(merged_d).mark_area(opacity=0.15, color="#dc2626").encode(
            x="시각:T",
            y=alt.Y(f"{col}_q25:Q", scale=y_scale),
            y2=f"{col}_q75:Q",
        )
        layers.insert(0, defect_band)

    if golden_batch_raw is not None and not golden_batch_raw.empty:
        merged_g = data[["Measurement_No", "시각"]].merge(golden_batch_raw, on="Measurement_No", how="left")
        golden_line = alt.Chart(merged_g).mark_line(
            strokeDash=[4, 3], strokeWidth=1.5, color="#059669", opacity=0.7
        ).encode(x="시각:T", y=alt.Y(f"{col}:Q", scale=y_scale))
        layers.insert(0, golden_line)

    if show_iqr:
        iqr_band = (
            alt.Chart(pd.DataFrame({"y1": [q1], "y2": [q3]}))
            .mark_rect(opacity=0.15, color="#8fd19e")
            .encode(y=alt.Y("y1:Q", scale=y_scale), y2="y2:Q")
        )
        layers.append(iqr_band)

    if ref_stats is not None:
        mean, std = ref_stats["mean"], ref_stats["std"]

        if show_sigma3:
            band_df = pd.DataFrame({"y1_3": [mean - 3 * std], "y2_3": [mean + 3 * std]})
            band3 = alt.Chart(band_df).mark_rect(opacity=0.10, color="#ff4b4b").encode(
                y=alt.Y("y1_3:Q", scale=y_scale), y2="y2_3:Q"
            )
            layers.append(band3)

        center_line = alt.Chart(pd.DataFrame({"y": [mean]})).mark_rule(
            color="gray", strokeDash=[2, 2]
        ).encode(y=alt.Y("y:Q", scale=y_scale))
        layers.append(center_line)

        # 넬슨룰은 옵션이 아니라 항상 계산·표시(팀 피드백). Rule 1이 더 심각하므로
        # 같은 점이 둘 다 걸리면 Rule 1로만 표시한다.
        z = (series - mean) / std
        rule1 = _nelson_rule1(z)
        rule5 = _nelson_rule5(z)
        data["_rule1"] = rule1.values
        data["_rule5"] = rule5.values & ~rule1.values
        data.loc[data["_rule1"], "넬슨룰 판정"] = "넬슨룰 1번 위반 (±3σ 밖, 급성 이상)"
        data.loc[data["_rule5"], "넬슨룰 판정"] = "넬슨룰 5번 위반 (조기 경고)"

        rule1_pts = data[data["_rule1"]]
        if len(rule1_pts):
            layers.append(
                alt.Chart(rule1_pts).mark_point(
                    shape="cross", size=140, color="#ff4b4b", strokeWidth=3
                ).encode(x="시각:T", y=alt.Y(f"{col}:Q", scale=y_scale))
            )

        rule5_pts = data[data["_rule5"]]
        if len(rule5_pts):
            layers.append(
                alt.Chart(rule5_pts).mark_point(
                    shape="triangle-up", size=100, color="#fab42d", filled=True
                ).encode(x="시각:T", y=alt.Y(f"{col}:Q", scale=y_scale))
            )

    line = base.mark_line(color="#5b9bd5")
    selectors = base.mark_point(size=200).encode(
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip("시각:T", title="시각", format="%H:%M:%S"),
            alt.Tooltip("측정순번:N", title="측정 순번(전체 중)"),
            alt.Tooltip(f"{col}:Q", title=var_name, format=".2f"),
            alt.Tooltip("넬슨룰 판정:N", title="넬슨룰 판정"),
        ],
    ).add_params(hover)
    hover_points = base.mark_point(color="#5b9bd5", size=70).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    hover_rule = base.mark_rule(color="gray", strokeDash=[3, 3]).encode(
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
    ).transform_filter(hover)

    layers += [line, selectors, hover_points, hover_rule]

    if show_min:
        min_point = (
            alt.Chart(min_row)
            .mark_point(color="#ff4b4b", filled=True, size=70)
            .encode(x="시각:T", y=alt.Y(f"{col}:Q", scale=y_scale))
        )
        layers.append(min_point)

    # 제목에서 방향·기준값 괄호 설명 제거(팀 피드백) — 변수명만 표시.
    title = f"{var_name} 관리도"
    chart = layers[0]
    for l in layers[1:]:
        chart = chart + l
    return chart.properties(height=320, title=title)


def shap_contribution_chart(contrib: pd.DataFrame, top_n: int = 6) -> alt.Chart:
    """SHAP 이탈진단 패널 — 선택 LOT에 대해 채택 모델(XGBoost 12F)이 어느 피처를
    근거로 판정했는지 가로 막대로 표시. 빨강(양수)=불량 쪽으로 민 피처,
    파랑(음수)=정상 쪽으로 민 피처("위험"이라는 말을 붙이면 정상 쪽까지 위험하게
    읽혀 어색하다는 피드백으로 뗌, 2026-09). |기여도| 큰 순서로 top_n개만 표시
    (12개 전부 보여주면 인사이트 규칙기반 목록과 중복감이 커서 요약만, 2026-09).
    model_data.get_shap_contributions()가 이미 |shap| 내림차순으로 정렬해 준다."""
    data = contrib.head(top_n).copy()
    data["방향"] = data["shap"].map(lambda v: "불량 쪽" if v > 0 else "정상 쪽")
    order = data["label"].tolist()

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="SHAP 기여도 (모델 판정에 미친 영향)"),
            # 이전엔 properties(height=28*n+40)로 전체 높이를 어림잡아 계산했는데,
            # 실제 렌더링에서 범례+축 영역을 뺀 실질 막대 영역이 너무 좁아져서
            # Vega가 라벨 겹침으로 판단해 절반가량을 자동으로 숨겨버리는 문제가
            # 있었음(막대는 6개 다 그려지는데 라벨은 3개만 보임, 실사용자 리포트로
            # 발견, 2026-09). alt.Step으로 "행 하나당 고정 픽셀"을 직접 지정해서
            # 범례/축과 무관하게 막대 영역 자체가 항상 충분히 확보되게 함.
            y=alt.Y(
                "label:N", title=None, sort=order,
                axis=alt.Axis(labelOverlap=False),
            ),
            color=alt.Color(
                "방향:N",
                scale=alt.Scale(domain=["불량 쪽", "정상 쪽"], range=["#ff4b4b", "#5b9bd5"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="피처"),
                alt.Tooltip("value:Q", title="값", format=".3f"),
                alt.Tooltip("shap:Q", title="SHAP 기여도", format=".3f"),
            ],
        )
        .properties(height=alt.Step(32))
    )
    return chart


def confidence_distribution_chart(df: pd.DataFrame, category: str) -> alt.Chart:
    """이미지 데이터 페이지 — 카테고리(정상/불량)별 판정 확률(불량일 확률) 분포.
    "확신도"는 실제로는 모델이 예측한 클래스의 확률값일 뿐인데 사람에게 과도한
    신뢰를 주는 단어로 느껴질 수 있어, "판정 확률"로 용어 변경(팀 피드백).
    X축: 번호(파일명 숫자), Y축: 불량일 확률(%), 기준선: 50%.
    df는 image_data.load_confidence_scores() 결과를 category로 미리 필터링한 것."""
    point_color = "#ff4b4b" if category == "불량" else "#5b9bd5"

    base = alt.Chart(df).encode(
        x=alt.X("sort_key:Q", title="번호"),
        y=alt.Y("defect_prob:Q", title="불량일 확률(%)", scale=alt.Scale(domain=[0, 100])),
    )
    points = base.mark_circle(size=35, opacity=0.6, color=point_color).encode(
        tooltip=[
            alt.Tooltip("label:N", title="이미지"),
            alt.Tooltip("defect_prob:Q", title="불량 확률(%)", format=".2f"),
        ]
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y": [50]}))
        .mark_rule(color="gray", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return (points + threshold).properties(
        height=260, title=f"{category} 이미지 판정 확률 분포 (기준선: 50%)"
    )
