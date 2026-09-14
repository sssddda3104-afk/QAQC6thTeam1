import streamlit as st
import pandas as pd
import hashlib
import sensor_data as sd
import model_data as md
import chart_helpers as ch
import image_data as imgd
import ai_report

st.set_page_config(page_title="크로메이트 도금 공정 모니터링", layout="wide")

# 표(st.dataframe)의 "열 표시/숨기기·검색·전체화면" 아이콘만 숨긴다.
# :has()로 "열 표시/숨기기" 버튼이 있는 툴바(=표 전용 툴바)만 골라 범위를 좁혀서,
# 그래프(차트) 쪽 툴바는 건드리지 않는다. 열 너비 드래그(리사이즈)와
# 헤더 클릭 정렬은 이 버튼들과 무관한 별도 동작이라 그대로 남는다.
st.markdown(
    """
    <style>
    div[data-testid="stElementToolbar"]:has(button[aria-label="Show/hide columns"]) {
        display: none;
    }

    /* ── KPI 카드 (예전 프로젝트 디자인 참고 반영, 2026-09) ──
       원래 흰 배경+회색 테두리였는데, 참고 이미지의 "왼쪽 포인트 컬러 바 + 옅은
       배경" 카드 스타일을 채용 — 다만 참고 이미지는 다크 테마(짙은 배경+청록
       글자)였고, 이 대시보드는 라이트 테마라 배경만 연한 푸른색으로 가져오고
       글자색은 검정으로 유지(요청 반영). "측정값·비율류는 초록 포인트 컬러"
       규칙은 이제 전부 같은 파랑 카드로 통일되면서 의미가 없어져 제거함.
       처음엔 :has(div[data-testid="stMetric"])로 "지표가 든 카드"를 판별했는데,
       :has()를 지원 안 하는 브라우저(구버전 등)에서는 이 규칙 전체가 조용히
       무시되어 스타일이 하나도 안 먹는 문제가 실사용자 리포트로 발견됨(2026-09).
       그래서 :has() 없이, 이미 각 카드에 준 고유 key(kpi-*/accent-*)를 직접
       선택자로 써서 브라우저 호환성 문제 없이 동작하게 함("불량 발생 목록"
       스크롤 상자에 쓴 것과 같은 st-key- 클래스 선택 패턴).
       height를 고정값으로 통일(min-height였을 때는 라벨 줄바꿈 여부에 따라
       카드마다 1~2px 차이가 남아 있었음 — 고정 height로 완전히 맞춤). */
    div[class*="st-key-kpi-"], div[class*="st-key-accent-"] {
        background: #eaf3fb !important;
        border: 1px solid #cfe3f5 !important;
        border-left: 4px solid #4a90d9 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(16, 24, 20, 0.06), 0 1px 2px rgba(16, 24, 20, 0.04);
        height: 96px !important;
        flex: 0 0 96px !important;
        align-self: stretch !important;
        padding: 0 !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }
    div[class*="st-key-kpi-"] div[data-testid="stElementContainer"],
    div[class*="st-key-accent-"] div[data-testid="stElementContainer"] {
        padding: 10px 16px !important;
    }
    div[class*="st-key-kpi-"] label[data-testid="stMetricLabel"],
    div[class*="st-key-accent-"] label[data-testid="stMetricLabel"] {
        color: #111827 !important;
        font-size: 0.82rem !important;
    }
    div[class*="st-key-kpi-"] [data-testid="stMetricValue"],
    div[class*="st-key-accent-"] [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    /* st.metric 위젯 전반 — 라벨/값 둘 다 확실히 가운데 정렬
       (라벨은 Streamlit 기본 CSS가 text-align:left를 더 구체적인 선택자로
       지정해서, 위의 일반 규칙만으로는 안 먹었음 — !important로 명시) */
    [data-testid="stMetric"] {
        text-align: center;
    }
    [data-testid="stMetric"] > div {
        justify-content: center;
    }
    [data-testid="stMetricLabel"] {
        justify-content: center !important;
        text-align: center !important;
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    [data-testid="stMetricLabel"] > div {
        text-align: center !important;
        width: 100%;
    }

    /* 디자인 참고(TFinder) 반영 — 사이드바에 옅은 톤을 줘서 본문과 구분되게 하고,
       페이지 전환 메뉴(세그먼트 버튼)를 감싸는 파스텔 배경 바를 추가해서
       참고 디자인의 nav bar 느낌을 살림(2026-09). */
    section[data-testid="stSidebar"] {
        background-color: #f2f7f4;
    }
    div[data-testid="stSegmentedControl"] {
        background: #e3f0e8;
        border-radius: 12px;
        padding: 6px;
    }

    /* "불량 발생 목록" 스크롤 상자(key="defect-list-box")를 목록 컬럼 전체 폭이
       아니라 내용에 맞는 폭으로 줄여서, 스크롤바가 "조회" 버튼과 멀리 떨어지지
       않고 바로 옆에 오게 함(요청 반영, 2026-09). */
    div[class*="st-key-defect-list-box"] {
        max-width: 420px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _load_data():
    df = sd.load_raw()
    lots = sd.get_lot_list(df)
    lot_summary = md.load_lot_summary()
    progress_cp = md.load_progress_checkpoint()
    timeseries = md.load_timeseries()
    ref_stats = sd.get_normal_reference_stats(df)
    golden_batch = md.get_golden_batch_profile(timeseries)
    defect_range = md.get_defect_batch_range(timeseries)
    golden_batch_raw = md.get_golden_batch_profile_raw(timeseries)
    defect_range_raw = md.get_defect_batch_range_raw(timeseries)
    return (
        df, lots, lot_summary, progress_cp, timeseries, ref_stats,
        golden_batch, defect_range, golden_batch_raw, defect_range_raw,
    )


# st.spinner()의 기본 스피너는 작고 텍스트도 옅어서, 실제로는 뜨는데도(프레임 단위로
# 캡처해서 확인함 — 뜨긴 뜨지만 로딩이 워낙 빨라서 스쳐 지나가 놓치기 쉬웠음) "안 뜬다"고
# 느껴질 수 있었음. 화면 중앙에 큼직한 회전 고리 + "Loading..."을 직접 그려서 훨씬 눈에
# 띄게 만들고, st.empty()로 로딩 끝나면 확실히 지운다(2026-09, 요청 반영).
_loading_slot = st.empty()
_loading_slot.markdown(
    """
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; padding: 72px 0;">
      <div style="width:52px; height:52px; border:5px solid #e5e9e7;
                  border-top-color:#059669; border-radius:50%;
                  animation:_spin 0.8s linear infinite;"></div>
      <div style="margin-top:18px; font-size:1.05rem; color:#4b5563;">Loading...</div>
    </div>
    <style>@keyframes _spin { to { transform: rotate(360deg); } }</style>
    """,
    unsafe_allow_html=True,
)
(
    df, lots, lot_summary, progress_cp, timeseries, ref_stats,
    golden_batch, defect_range, golden_batch_raw, defect_range_raw,
) = _load_data()
_loading_slot.empty()


def _goto_lot(date_str: str, lot_num: int) -> None:
    """"조회" 버튼(불량 발생 목록)의 on_click 콜백 — 날짜/LOT 선택자를 그 값으로 이동시킨다.
    ⚠ 반드시 on_click 콜백으로 해야 한다: 버튼을 `if st.button(...): st.session_state.sel_date = ...`
    식으로 스크립트 본문 안에서 직접 처리하면, 이 코드가 실행되는 시점(본문 아래쪽)에는
    이미 사이드바의 sel_date/sel_lot 위젯이 그 위(사이드바)에서 먼저 인스턴스화된 뒤라서
    "StreamlitWidgetAlreadyInstantiatedError: 위젯 인스턴스화 후에는 그 key의
    session_state를 못 바꾼다"는 예외가 난다(실제로 겪은 버그 — 히트맵 클릭 기능도
    같은 이유로 안 먹혔었다). on_click 콜백은 스크립트 본문이 다시 실행되기 *전*,
    즉 사이드바 위젯이 인스턴스화되기 전에 미리 실행되므로 이 제약에 걸리지 않는다."""
    st.session_state.sel_date = date_str
    st.session_state.sel_lot = str(lot_num)


def _notice(text: str, kind: str = "info") -> None:
    """참고한 타사 대시보드(TFinder)처럼, 중요 안내문을 파스텔 컬러 박스로 강조.
    기본 st.info()/st.caption()보다 눈에 잘 들어오고, 색으로 "좋은 소식(info)"과
    "주의/미채택 등(warn)"을 구분해서 전달한다(2026-09, 디자인 참고 반영).
    kind: "info"(연초록, 설명·공지용) 또는 "warn"(연노랑, 주의·제약사항용)."""
    palette = {
        "info": ("#eafaf1", "#1f7a4d", "#bfe6cf"),
        "warn": ("#fdf6e3", "#8a6d1f", "#f0e2ab"),
    }
    bg, fg, border = palette.get(kind, palette["info"])
    st.markdown(
        f'<div style="background:{bg};color:{fg};border:1px solid {border};'
        f'border-radius:8px;padding:12px 16px;margin:4px 0 12px;font-size:0.92rem;'
        f'line-height:1.5;">{text}</div>',
        unsafe_allow_html=True,
    )


def _ai_report_section(label: str, cache_key: str) -> None:
    """AI 자동 리포트 4개 하위 섹션(전체 요약/관리도/모델/SHAP)이 공통으로 쓰는
    표시 전용 UI — 개별 생성 버튼은 없음(요청 반영, 2026-09: "전체 한번에 생성"
    버튼 하나로만 채우고, 이 함수는 session_state에 이미 캐싱된 결과가 있으면
    그것만 보여준다). 아직 생성 전이면 안내 캡션만 표시."""
    with st.container(border=True):
        st.caption(f"🤖 {label} (OpenAI gpt-4o-mini)")
        if st.session_state.get(cache_key):
            st.markdown(st.session_state[cache_key])
        else:
            st.caption("아직 생성되지 않았습니다 — 위 \"AI 자동 리포트 생성\" 버튼을 눌러주세요.")



def _shap_insight(contrib: pd.DataFrame, top_n: int = 3) -> str:
    """SHAP 기여도 기반 인사이트 문장 — 기존 이탈률 규칙 기반 인사이트를 대체함
    (2026-09). SHAP이 이탈률 3개를 포함한 12개 피처 전부를 이미 반영하고, 실제
    모델이 판정에 쓴 근거와 다를 수 있는 규칙(이탈률 기준)보다 신뢰도가 높다고
    판단해 규칙기반 문장은 빼고 SHAP 결과를 그대로 문장으로 풀어서 보여준다.
    단, "이 값을 낮추면 정상화된다"는 식의 조정 방향은 말하지 않는다 — 트리
    모델은 피처-결과 관계가 항상 단조롭다는 보장이 없어, SHAP 기여도만으로
    그런 인과적 조정 방향을 단정하는 것은 과장이 될 수 있기 때문."""
    if contrib.empty:
        return "표시할 SHAP 결과가 없습니다."
    top = contrib.iloc[0]
    top_dir = "불량" if top["shap"] > 0 else "정상"
    lines = [
        f"이번 LOT은 <b>{top['label']}</b>(값 {top['value']:.3g})이 모델 판정에 가장 크게 "
        f"기여했습니다 (SHAP 기여도 {top['shap']:+.3f}, <b>{top_dir}</b> 쪽으로 작용) — "
        f"채택 모델(XGBoost 12F)이 이 LOT을 판정할 때 실제로 가장 많이 참고한 값입니다.",
        "<br>피처별 상세(|기여도| 큰 순):",
    ]
    for i, row in contrib.head(top_n).reset_index(drop=True).iterrows():
        d = "불량" if row["shap"] > 0 else "정상"
        lines.append(
            f"&nbsp;&nbsp;{i+1}. <b>{row['label']}</b> — 값 {row['value']:.3g}, "
            f"SHAP 기여도 {row['shap']:+.3f} ({d} 쪽)"
        )
    return "<br>".join(lines)


def _kpi_card(col, label: str, value: str, key: str, help: str | None = None) -> None:
    """카드 하나 생성. 모든 카드가 같은 연한 푸른색 스타일(참고 이미지 디자인
    반영, 2026-09)이라 key의 "accent-" 접두어는 이제 스타일에 영향 없음 —
    기존 코드와의 호환을 위해 접두어 자체는 그대로 남겨둠.
    help: 라벨 오른쪽에 물음표(?) 아이콘을 붙여서, 마우스를 올리면 설명이 뜨게 함
    (st.metric 내장 기능 — 튜터 피드백으로 "아래에 길게 설명하지 말고 도움말(?)로"
    요청받아 반영, 2026-09). None이면 물음표 없이 기존과 동일."""
    with col.container(border=True, key=key):
        st.metric(label, value, help=help)


def _clicked_point_idx(chart_state, n: int) -> int | None:
    """combined_zscore_chart의 "point_select" 클릭 결과에서 lot_ts 행 번호(정수)를
    꺼낸다. 타임스탬프 대신 정수 인덱스를 왕복시키므로 파싱 오류 여지가 없다.
    클릭된 점이 없거나 범위를 벗어나면 None을 반환 — 호출부에서 마지막 값으로 대체한다."""
    if not chart_state or not getattr(chart_state, "selection", None):
        return None
    points = chart_state.selection.get("point_select")
    if not points:
        return None
    raw = points[0].get("_idx")
    if raw is None:
        return None
    try:
        idx = int(raw)
    except (ValueError, TypeError):
        return None
    if not (0 <= idx < n):
        return None
    return idx


# 파생변수 표(_render_variable_table)·이탈률 진행바(_render_deviation_rates)는
# 한때 팀 피드백으로 "있어야 할 의미가 없다"고 제거했었으나, 지금은 브레인스토밍 단계라
# 나중에 분석가/현장직 탭으로 나눌 때 고를 수 있게 일단 복구해둠(2026-09).
def _render_variable_table(var_table: pd.DataFrame) -> None:
    """파생변수 표를 이탈률(%)·방향 컬럼은 빼고 표시 + 나머지 숫자 컬럼(표준편차/최솟값/IQR)에
    정상↔불량 두 기준점 사이 위치로 색칠(요청 반영, 2026-09 — 기존엔 지금 표에 보이는
    3개 행끼리만 비교하는 상대 히트맵(Oranges, 컬럼별 정규화)이었는데, "정상에 가까우면
    파랑, 불량에 가까우면 빨강"으로 바꿔달라는 요청). 정상/불량 기준값은 726 LOT 전체의
    그룹별 중앙값(model_data.get_defect_lean_reference)이라 이 표에 안 보이는 다른
    LOT들과 비교한 절대적 위치를 보여준다 — 표에 보이는 3행끼리의 상대 비교가 아님.
    "변수"(pH/온도/전압)는 행 인덱스로 옮겨서 실제 테이블의 행 헤더(th)가 되게 한다.
    이탈률(%)·방향은 표에서 빼는 대신 바로 아래 진행바로 따로 보여준다(_render_deviation_rates).
    행 헤더(변수명)·열 헤더(표준편차 등)에 배경색을 줘서 "라벨"과 "실제 값"이 확실히
    구분되게 함(요청 반영, 2026-09) — 라이트 테마 전환 때 빠졌던 헤더 스타일을 복구."""
    heatmap_cols = [c for c in ["표준편차", "최솟값", "IQR"] if c in var_table.columns]
    display_cols = [c for c in var_table.columns if c not in ("이탈률(%)", "방향")]
    display_df = var_table[display_cols].set_index("변수")

    fmt = {"표준편차": "{:.3f}", "최솟값": "{:.2f}", "IQR": "{:.3f}"}
    if "현재값" in display_df.columns:
        fmt["현재값"] = "{:.2f}"

    # 표의 "변수"(pH/온도/전압) 행 라벨 ↔ 핸드오프 CSV의 피처 접두어(pH/Temp/Voltage),
    # "표준편차/최솟값/IQR" 열 라벨 ↔ 피처 접미어(_std/_min/_IQR) 매핑.
    var_to_code = {"pH": "pH", "온도": "Temp", "전압": "Voltage"}
    col_to_stat = {"표준편차": "std", "최솟값": "min", "IQR": "IQR"}
    ref = md.get_defect_lean_reference()

    # SHAP 이탈진단 차트와 같은 색(#5b9bd5 파랑=정상 쪽 / #ff4b4b 빨강=불량 쪽)을
    # 그대로 써서 두 시각화의 색 언어를 통일함(요청 반영, 2026-09) — matplotlib
    # 콜맵(RdBu_r)을 alpha blending으로 얹었더니 탁한 자주색으로 보인다는 피드백으로,
    # 직접 3점(파랑→흰색→빨강) 선형보간해 중간(=거의 안 치우침)은 흰색에 가깝고
    # 끝으로 갈수록 SHAP과 동일한 선명한 색이 나오게 함.
    _BLUE, _WHITE, _RED = (91, 155, 213), (255, 255, 255), (255, 75, 75)

    def _lean_hex(t: float) -> str:
        t = min(max(t, 0.0), 1.0)
        if t <= 0.5:
            frac = t / 0.5
            c0, c1 = _BLUE, _WHITE
        else:
            frac = (t - 0.5) / 0.5
            c0, c1 = _WHITE, _RED
        rgb = tuple(round(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _lean_colors(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for col in heatmap_cols:
            stat = col_to_stat[col]
            for var_label in data.index:
                code = var_to_code.get(var_label)
                feat = f"{code}_{stat}" if code else None
                if not feat or feat not in ref:
                    continue
                lo, hi = ref[feat]  # (정상 중앙값, 불량 중앙값)
                v = data.loc[var_label, col]
                t = 0.5 if hi == lo else (v - lo) / (hi - lo)
                styles.loc[var_label, col] = f"background-color: {_lean_hex(t)}"
        return styles

    styled = (
        display_df.style
        .apply(_lean_colors, axis=None)
        .format(fmt)
        .set_table_styles([
            {"selector": "th.row_heading", "props": [
                ("background-color", "#dcece2"), ("color", "#14532d"),
                ("font-weight", "700"),
            ]},
            {"selector": "th.col_heading", "props": [
                ("background-color", "#f0f3f1"), ("color", "#374151"),
                ("font-weight", "600"),
            ]},
        ])
    )
    st.table(styled)


def _render_deviation_rates(var_table: pd.DataFrame) -> None:
    """이탈률(%)만 변수별 진행바로 따로 표시. LOT 하나(그 자체의 시계열) 안에서
    측정값이 기준을 벗어난 비율이라는 걸 캡션으로 같이 안내한다."""
    st.caption("이탈률 — 이 LOT 자체 측정값 중 기준을 벗어난 비율")
    for _, row in var_table.iterrows():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write(f"**{row['변수']}** ({row['방향']})")
        with c2:
            st.progress(min(row["이탈률(%)"] / 100, 1.0), text=f"{row['이탈률(%)']}%")


def _control_chart_options() -> dict:
    """관리도에 표시할 오버레이를 고르는 체크박스 3개(넬슨룰 1·5는 이제 옵션이 아니라
    항상 자동 적용되어 여기서 빠짐 — 팀 피드백. "팀 기준선"(구 이탈 기준선)도 넬슨룰
    자동판정과 역할이 겹쳐 불필요하다는 팀 피드백으로 제거함). 양쪽(과거뷰/실시간뷰)
    관리도가 같은 설정을 공유하도록 session_state(key)에 저장한다.
    체크박스 자신이 참조하는 값을 스스로 되써버리는 self-reference 패턴은
    사이드바 메뉴 라디오에서 겪었던 것과 같은 버그(위젯 정체성 꼬임)를 유발하므로,
    value= 수동 재할당 대신 key=만 써서 Streamlit이 알아서 관리하게 한다.
    옵션 패널 자체도 기본은 접혀 있고 필요한 사람만 펼쳐서 보게 expander로 감쌈(팀 피드백)."""
    defaults = {
        "cc_show_sigma3": False,
        "cc_show_iqr": False, "cc_show_min": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    with st.expander("관리도 표시 옵션", expanded=False):
        o1, o2, o3 = st.columns(3)
        o1.checkbox("3σ 밴드", key="cc_show_sigma3")
        o2.checkbox("IQR 음영", key="cc_show_iqr")
        o3.checkbox("최솟값 표시", key="cc_show_min")
    return {
        "show_sigma3": st.session_state.cc_show_sigma3,
        "show_iqr": st.session_state.cc_show_iqr,
        "show_min": st.session_state.cc_show_min,
    }

# ── 세션 상태 초기화 ──────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🧪 센서 데이터"
# "보기 선택"(view_mode) 토글은 폐지 — 튜터 피드백으로 한 화면에 정보가 너무 많다는
# 지적을 반영해, 본문 상단에 크롬 탭 스타일 탭바(sensor_tab)로 재구성함(2026-09).
# "선택 로트 진행 분석"은 별도 토글이 아니라 "로트 분석" 탭 그 자체가 됨.
if "sensor_tab" not in st.session_state:
    st.session_state.sensor_tab = "데이터"

# 갤러리에서 "조회" 버튼을 누르면 위젯이 이미 그려진 뒤라 session_state를
# 직접 못 바꾸므로(StreamlitWidgetAlreadyInstantiatedError), 대기 상태에
# 넣어뒀다가 위젯을 그리기 "전"인 다음 런 시작 시점에 반영한다.
if "pending_img_category" in st.session_state:
    st.session_state["img_category"] = st.session_state.pop("pending_img_category")
if "pending_selected_img_label" in st.session_state:
    st.session_state["selected_img_label"] = st.session_state.pop("pending_selected_img_label")

# ── 사이드바: 메뉴 ──────────────────────────────
with st.sidebar:
    st.markdown("### 메뉴")
    # index를 session_state.page로부터 계산해서 넣으면, 이 위젯 자신이 그
    # session_state.page를 갱신하는 self-reference가 생겨 위젯 정체성(auto key)이
    # 매 런마다 바뀌어버리는 문제가 있었다(예: "외관 검사"에서 "센서 데이터"로
    # 못 돌아가는 버그). key로 직접 session_state에 묶어서 index 계산을 없앤다.
    # st.radio(원형 버튼) 대신 st.segmented_control(버튼형, 선택된 항목이
    # 배경색으로 강조됨 — Claude 사이드바 탭과 유사한 느낌)로 교체.
    # required=True로 항상 하나는 선택된 상태를 유지(탭처럼 동작).
    st.segmented_control(
        "메뉴 선택",
        ["🧪 센서 데이터", "🔍 외관 검사"],
        label_visibility="collapsed",
        key="page",
        required=True,
        width="stretch",
    )
    page = st.session_state.page

    st.divider()

    if page == "🧪 센서 데이터":
        st.markdown("**1. 기간 선택**")
        min_d, max_d = df["date"].min(), df["date"].max()
        date_input_val = st.date_input(
            "조회 기간", value=(min_d, max_d), min_value=min_d, max_value=max_d,
            label_visibility="collapsed",
        )
        # 범위(튜플) / 단일 날짜 둘 다 지원 — 하루만 선택해도 동작해야 함
        if isinstance(date_input_val, tuple) and len(date_input_val) == 2:
            start_d, end_d = date_input_val
        elif isinstance(date_input_val, tuple) and len(date_input_val) == 1:
            start_d = end_d = date_input_val[0]
        else:
            start_d = end_d = date_input_val
        st.session_state.period = (start_d, end_d)

        st.markdown("**2. LOT 선택**")
        period_lots = lots[(lots["date"] >= start_d) & (lots["date"] <= end_d)].copy()
        # 기존엔 "날짜 · Lot 번호"를 합쳐 하나의 드롭다운으로 골랐는데, 날짜/LOT
        # 번호 두 단계로 나눠달라는 요청 반영 — 날짜는 선택한 기간 안의 날짜만 보여준다.
        period_dates = sorted(period_lots["date"].unique().tolist())
        default_date = pd.Timestamp("2021-09-08").date()
        if not period_dates:
            st.caption("해당 기간에는 데이터가 없습니다.")
            st.session_state.selected_lot_label = None
            st.session_state.selected_lot = None
            selected_date = None
        else:
            date_options = [str(d) for d in period_dates]
            default_date_str = str(default_date) if default_date in period_dates else date_options[0]
            # key="sel_date"/"sel_lot"로 고정해서 히트맵 클릭 시(본문 쪽 코드) 여기
            # session_state를 직접 갱신 + st.rerun()으로 이 선택자들을 그 값으로
            # 바로 이동시킬 수 있게 한다(팀 피드백). index= 대신 session_state를
            # 먼저 채워두는 이유는, 위젯에 key와 index를 같이 주면 이미 값이 있는
            # 이후 실행에서 index가 무시되어 외부에서 갱신한 값과 계속 안 맞기 때문.
            if "sel_date" not in st.session_state or st.session_state.sel_date not in date_options:
                st.session_state.sel_date = default_date_str
            selected_date_str = st.selectbox("날짜 선택", options=date_options, key="sel_date")
            selected_date = pd.Timestamp(selected_date_str).date()

            date_lots = period_lots[period_lots["date"] == selected_date]
            lot_num_options = ["전체"] + [str(n) for n in sorted(date_lots["lot"].tolist())]
            default_lot_str = "20" if selected_date == default_date and "20" in lot_num_options else "전체"
            if "sel_lot" not in st.session_state or st.session_state.sel_lot not in lot_num_options:
                st.session_state.sel_lot = default_lot_str
            selected_lot_str = st.selectbox("LOT(슬롯) 선택", options=lot_num_options, key="sel_lot")
            if selected_lot_str == "전체":
                st.session_state.selected_lot_label = "전체"
                st.session_state.selected_lot = None
            else:
                st.session_state.selected_lot_label = f"{selected_date} · Lot {selected_lot_str}"
                st.session_state.selected_lot = (selected_date, int(selected_lot_str))

        st.markdown("**3. 주요변수 선택**")
        show_ph = st.checkbox("pH", value=True)
        show_temp = st.checkbox("온도", value=True)
        show_volt = st.checkbox("전압", value=True)
        st.session_state.show_vars = {"pH": show_ph, "온도": show_temp, "전압": show_volt}

    else:  # 외관 검사
        st.markdown("**1. 대분류 선택**")
        img_category = st.radio(
            "대분류", ["정상", "불량"], index=1, label_visibility="collapsed", key="img_category"
        )

        st.markdown("**2. 번호 선택**")
        samples_df = imgd.get_samples_df()
        cat_samples = samples_df[samples_df["대분류"] == img_category]["번호"].tolist()
        if cat_samples:
            selected_img_label = st.selectbox(
                "번호 선택", options=cat_samples, label_visibility="collapsed", key="selected_img_label"
            )
        else:
            st.caption(f"'{img_category}' 분류의 샘플 이미지가 아직 없습니다.")
            selected_img_label = None

st.title("크로메이트 도금 공정 모니터링")

# ── 메인 영역 ──────────────────────────────
if st.session_state.page == "🧪 센서 데이터":
    # 크롬 탭 스타일 탭바(요청 반영, 2026-09) — 한 화면에 정보가 너무 많다는
    # 튜터 피드백으로 "데이터"/"로트 분석"/"모델·인사이트" 3개로 분리.
    # st.tabs는 코드로 강제 전환이 안 되므로(히트맵 옆 "조회" 버튼 같은 프로그래매틱
    # 이동 패턴과 충돌), 사이드바 메뉴와 같은 segmented_control 방식을 그대로 씀.
    st.segmented_control(
        "탭 선택",
        ["데이터", "로트 분석", "모델·인사이트"],
        label_visibility="collapsed",
        key="sensor_tab",
        required=True,
    )

    if st.session_state.sensor_tab == "로트 분석":
        sel = st.session_state.get("selected_lot")
        lot_label = st.session_state.get("selected_lot_label")
        if not sel:
            if lot_label == "전체":
                st.info("선택 로트 진행 분석은 LOT 단위로 동작합니다. 사이드바에서 특정 LOT을 선택해주세요.")
            else:
                st.warning("사이드바에서 LOT을 선택해주세요.")
        else:
            lot_date, lot_slot = sel

            st.caption(f"진행 분석 대상: {lot_label}")
            progress = st.slider("진행률", min_value=0, max_value=100, value=60, step=1)

            lot_ts = md.get_lot_timeseries(timeseries, lot_date, lot_slot)
            total_points = len(lot_ts)

            if total_points == 0:
                st.warning("해당 LOT의 시계열 데이터가 없습니다.")
            else:
                # 그래프·표는 실제 진행률(1% 단위)에 맞춰 즉시 계산
                upto = max(1, round(total_points * progress / 100))

                # AI 참고 위험도는 팀이 사전계산한 체크포인트(20/30/.../100%)에서만
                # 나온다 — 슬라이더가 그 사이 값이면 "이하의 가장 가까운 체크포인트" 기준으로 표시
                available = [p for p in md.PROGRESS_LEVELS if p <= progress]
                checkpoint = max(available) if available else None

                _notice(
                    "AI 참고 위험도(Early Warning)는 <b>최종 채택된 모델이 아닙니다.</b> "
                    "75~80% 구간까지도 안정성 기준(불량 9건 중 최소 7건 검출)을 만족하지 못해 "
                    "참고용으로만 제공됩니다. 이 값을 근거로 한 확정 판정은 하지 않습니다.",
                    kind="warn",
                )

                if checkpoint is None:
                    ew = None
                else:
                    ew = md.get_ew_oof(progress_cp, lot_date, lot_slot, checkpoint)

                show_vars = st.session_state.get("show_vars", {"pH": True, "온도": True, "전압": True})

                st.subheader(f"선택 시점({progress}%)까지의 그래프")
                # 참고 대시보드 스타일 반영: 그래프 옆에 진행률·AI 위험도 지표를 나열
                # (기존엔 그래프 위에 별도 지표 행으로 있었음 — 팀 피드백으로 나란히 배치).
                graph_col, side_col = st.columns([3, 1])
                with graph_col:
                    st.altair_chart(
                        ch.combined_zscore_chart(lot_ts.iloc[:upto], show_vars), width="stretch"
                    )
                with side_col:
                    st.metric("진행률", f"{progress}%")
                    ai_help = (
                        "채택된 최종 모델이 아니라 참고용 Early Warning 신호입니다.\n\n"
                        "정상 데이터 대비 위험 순위(percentile)를 **낮음/주의/높음** 3단계로 "
                        "표시합니다 — 75~80% 진행 구간까지도 안정성 기준(불량 9건 중 최소 "
                        "7건 검출)을 만족하지 못해 확정 판정에는 쓰지 않습니다."
                    )
                    alert_help = (
                        "이 시점까지의 Early Warning 확률이, 해당 체크포인트에서 미리 정해둔 "
                        "경보 기준값(threshold)을 넘었는지 여부입니다.\n\n"
                        "**예**여도 확정 불량 판정이 아니라 \"더 주의 깊게 볼 필요가 있다\"는 "
                        "참고 신호일 뿐입니다."
                    )
                    if checkpoint is None or not ew:
                        st.metric("AI 참고 위험도", "데이터 부족", help=ai_help)
                        st.metric("경보 기준 초과 여부", "-", help=alert_help)
                    else:
                        status = md.risk_status_label(ew["risk_percentile"])
                        st.metric("AI 참고 위험도", status, help=ai_help)
                        st.metric(
                            "경보 기준 초과 여부", "예" if ew["threshold_alert"] else "아니오",
                            help=alert_help,
                        )
                if checkpoint is not None and ew and checkpoint != progress:
                    st.caption(f"※ AI 참고 위험도는 최근 체크포인트인 {checkpoint}% 시점 기준입니다.")

                st.subheader("관리도")
                cc_opts = _control_chart_options()
                st.caption("점선 = 정상 LOT 평균 궤적")
                for var_name in ["pH", "온도", "전압"]:
                    if show_vars.get(var_name, True):
                        st.altair_chart(
                            ch.variable_control_chart(
                                lot_ts, var_name, ref_stats=ref_stats[var_name], upto=upto,
                                golden_batch_raw=golden_batch_raw,
                                **cc_opts
                            ),
                            width="stretch",
                        )
    elif st.session_state.sensor_tab == "데이터":
        start_d, end_d = st.session_state.get("period", (df["date"].min(), df["date"].max()))
        period_df = df[(df["date"] >= start_d) & (df["date"] <= end_d)]

        kpi = sd.get_overview_kpi(period_df)
        ooc = md.get_out_of_control_rate(timeseries, start_d, end_d)
        trend = md.get_recent_defect_rate_trend(df, end_d)

        def _fmt(value, suffix=""):
            return "-" if value is None else f"{value}{suffix}"

        def _fmt_trend(delta):
            if delta is None:
                return "-"
            if delta > 0:
                return f"▲{delta}%p"
            if delta < 0:
                return f"▼{abs(delta)}%p"
            return "변화없음"

        st.subheader("데이터 조회")
        c1, c2, c3, c4 = st.columns(4)
        _kpi_card(c1, "조회 기간", f"{kpi['조회 기간(일)']}일", key="kpi-period")
        _kpi_card(c2, "총 LOT 수", f"{kpi['총 LOT 수']}건", key="kpi-total-lot")
        _kpi_card(c3, "불량 LOT 수", f"{kpi['불량 LOT 수']}건", key="kpi-defect-lot")
        _kpi_card(c4, "불량률", _fmt(kpi["불량률(%)"], "%"), key="accent-defect-rate")

        c5, c6, c7, c8 = st.columns(4)
        _kpi_card(c5, "pH 관리이탈율", _fmt(ooc.get("pH"), "%"), key="accent-ooc-ph")
        _kpi_card(c6, "온도 관리이탈율", _fmt(ooc.get("온도"), "%"), key="accent-ooc-temp")
        _kpi_card(c7, "전압 관리이탈율", _fmt(ooc.get("전압"), "%"), key="accent-ooc-volt")
        _kpi_card(
            c8, "불량률 증감 (vs 지난 7일)", _fmt_trend(trend["변화"]), key="kpi-defect-trend",
            help=(
                "선택한 조회 기간의 마지막 7일과, 그 직전 7일의 불량률을 비교합니다.\n\n"
                "예: 조회 기간이 2021-09-06~2021-10-27이면 → "
                "2021-10-21~2021-10-27(최근 7일) vs 2021-10-14~2021-10-20(그 직전 7일)의 "
                "불량률 차이를 보여줍니다."
            ),
        )

        # 히트맵 영역과 목록 영역을 반반(1:1)으로 나눔 — 목록 쪽만 늘어나게 하는
        # 대신 화면 폭이 바뀌어도 두 영역이 같이 커지고 작아지게 하는 게 더 자연스럽다는
        # 피드백 반영(이전엔 목록만 늘어나는 CSS 트릭을 썼었는데 그건 걷어냄).
        # 제목도 컬럼 밖(위)에 하나로 두지 않고 각 컬럼 안에 넣어서, 두 제목이 같은
        # 줄에서 시작하고 그 아래 실제 내용(그래프/목록)도 상하 위치가 맞게 함.
        heat_col, list_col = st.columns([1, 1])
        with heat_col:
            st.markdown("**전체 LOT 중 불량 위치**")
            st.altair_chart(
                ch.defect_heatmap_chart(sd.get_lot_list(period_df)), width="content"
            )
            st.caption("회색 = 정상, 빨강 = 불량 (정상 칸은 마우스를 올려도 반응하지 않습니다)")
        with list_col:
            st.markdown("**불량 발생 목록**")
            defect_rows = sd.get_lot_list(period_df)
            defect_rows = defect_rows[defect_rows["error"] == 1].sort_values("date")
            with st.container(height=330, key="defect-list-box"):
                if defect_rows.empty:
                    st.caption("이 기간에는 불량이 없습니다.")
                else:
                    for _, row in defect_rows.iterrows():
                        row_date, row_lot = row["date"], int(row["lot"])
                        b1, b2 = st.columns([3, 2])
                        with b1:
                            st.write(f"📅 {row_date} · Lot {row_lot}")
                        with b2:
                            st.button(
                                "조회",
                                key=f"defect_list_goto_{row_date}_{row_lot}",
                                on_click=_goto_lot,
                                args=(str(row_date), row_lot),
                            )

        # "데이터" 탭 — 선택 LOT의 그래프·관리도까지만(모델판정/인사이트는 "모델·인사이트" 탭으로 분리, 2026-09).
        sel = st.session_state.get("selected_lot")
        lot_label = st.session_state.get("selected_lot_label")
        if sel:
            lot_date, lot_slot = sel
            show_vars = st.session_state.get("show_vars", {"pH": True, "온도": True, "전압": True})

            lot_ts = md.get_lot_timeseries(timeseries, lot_date, lot_slot)
            if not lot_ts.empty:
                st.subheader(f"시간-변수값 그래프 — {lot_label}")
                # 참고 대시보드 스타일 반영: 그래프 옆에 그 시점 값을 나열.
                # 점을 클릭하면 그 시점 값으로 갱신되고(팀 피드백), 클릭 전에는 마지막 값을 보여준다.
                # 자동판정은 사이드 패널이 아니라 그래프 "아래"에 별도 행으로 배치(팀 피드백).
                # 점선 = 정상 LOT 평균 궤적(참고 대시보드의 "Golden Batch" 개념을 우리 데이터로
                # 구현한 것 — 화면에는 "Golden Batch"라는 용어를 쓰지 않기로 함, 2026-09).
                graph_col, side_col = st.columns([3, 1])
                with graph_col:
                    chart_state = st.altair_chart(
                        ch.combined_zscore_chart(
                            lot_ts, show_vars, golden_batch=golden_batch
                        ),
                        on_select="rerun",
                        selection_mode=["point_select"],
                        key="zscore_chart_hist",
                        width="stretch",
                    )
                    st.caption("점선 = 정상 LOT 평균 궤적")
                with side_col:
                    clicked_idx = _clicked_point_idx(chart_state, len(lot_ts))
                    if clicked_idx is not None:
                        row = lot_ts.iloc[clicked_idx]
                        st.caption(f"선택 시점: {row['Timestamp'].strftime('%H:%M:%S')}")
                    else:
                        row = lot_ts.iloc[-1]
                        st.caption("점을 클릭하면 그 시점 값이 표시됩니다 (기본: 마지막 값)")
                    if show_vars.get("pH", True):
                        st.metric("pH", f"{row['pH']:.2f}")
                    if show_vars.get("온도", True):
                        st.metric("온도", f"{row['Temp']:.1f}℃")
                    if show_vars.get("전압", True):
                        st.metric("전압", f"{row['Voltage']:.1f}V")

                st.subheader("관리도")
                cc_opts = _control_chart_options()
                st.caption("점선 = 정상 LOT 평균 궤적")
                for var_name in ["pH", "온도", "전압"]:
                    if show_vars.get(var_name, True):
                        st.altair_chart(
                            ch.variable_control_chart(
                                lot_ts, var_name, ref_stats=ref_stats[var_name],
                                golden_batch_raw=golden_batch_raw,
                                **cc_opts
                            ),
                            width="stretch",
                        )
        else:
            if lot_label == "전체":
                st.info("특정 LOT을 선택하면 그래프·관리도를 볼 수 있습니다.")
            else:
                st.warning("사이드바에서 LOT을 선택해주세요.")

    elif st.session_state.sensor_tab == "모델·인사이트":
        # "모델" 탭과 "인사이트" 탭을 결국 하나로 합치기로 함(요청 반영, 2026-09) —
        # 파생변수 표를 규칙기반 인사이트·SHAP이 그대로 근거로 쓰기 때문에 나눠두면
        # 탭을 오가며 봐야 해서 오히려 불편하다는 점을 감안함.
        sel = st.session_state.get("selected_lot")
        lot_label = st.session_state.get("selected_lot_label")
        if sel:
            lot_date, lot_slot = sel
            show_vars = st.session_state.get("show_vars", {"pH": True, "온도": True, "전압": True})

            lot_ts = md.get_lot_timeseries(timeseries, lot_date, lot_slot)
            final_oof = md.get_final_oof(lot_summary, lot_date, lot_slot)
            if not lot_ts.empty:
                st.caption(f"선택 LOT: {lot_label}")

                if final_oof:
                    st.subheader("모델 판정 (완료 LOT · OOF 검증값)")
                    fc1, fc2, fc3 = st.columns(3)
                    # 원본 데이터(팀 핸드오프 CSV)는 "정상 위험"/"불량 위험"으로 돼 있는데,
                    # "위험"이 붙으면 정상 쪽도 위험한 것처럼 읽혀 어색하다는 피드백으로
                    # 화면 표시에서만 "위험"을 떼어 "정상"/"불량"으로 보여줌(2026-09).
                    pred = final_oof["prediction"].replace(" 위험", "")
                    pct = final_oof["risk_percentile"]
                    top_pct = (1 - pct) * 100
                    top_label = "상위 1% 이내" if top_pct < 1 else f"상위 {top_pct:.0f}%"
                    _kpi_card(
                        fc1, "모델 판정", pred, key="kpi-verdict",
                        help=(
                            "해당 LOT을 학습에서 제외한 OOF(out-of-fold) 검증 점수를 기준으로 "
                            "모델이 내린 판정입니다 — 과거 화면 표시용으로 평가 누수가 없습니다.\n\n"
                            "**정상**: 모델이 위험도가 낮다고 본 LOT\n\n"
                            "**불량**: 모델이 위험도가 높다고 본 LOT"
                        ),
                    )
                    _kpi_card(
                        fc2, "위험 순위", top_label, key="kpi-risk-rank",
                        help=(
                            "실제 불량 확률이 아니라, 정상 데이터 대비 상대적 위험 순위입니다.\n\n"
                            "예: **상위 1% 이내**는 지금까지 본 정상 LOT들 중 위험도가 가장 높은 "
                            "1% 안에 든다는 뜻이며, 숫자가 작을수록(상위일수록) 더 위험합니다."
                        ),
                    )
                    _kpi_card(fc3, "실제 결과", final_oof["actual_label"], key="kpi-actual-label")

                # 모델 정보 섹션(팀원 브레인스토밍 반영) — 지금까지 대시보드에서 안 쓰이던
                # dashboard_model_validation.csv를 그대로 가져다 씀. "채택"된 모델(완료 LOT
                # 판정용)만 메인으로 보여주고, Early Warning(미채택)은 캡션으로만 언급.
                # 모델 판정 바로 아래에 둬서 "이 판정을 낸 모델이 이거다"가 뒤이어 나오게 함(2026-09, 순서 변경).
                with st.expander("모델 정보", expanded=False):
                    mv = md.load_model_validation()
                    model_row = mv[mv["Decision"] == "채택"].iloc[0]
                    mi1, mi2, mi3, mi4 = st.columns(4)
                    _kpi_card(mi1, "사용 모델", "XGBoost 12F", key="kpi-model-name")
                    _kpi_card(mi2, "PR-AUC", f"{model_row['PR_AUC_Mean']:.2f}", key="accent-model-prauc")
                    _kpi_card(mi3, "Recall (평균)", f"{model_row['Recall_Mean']*100:.1f}%", key="accent-model-recall")
                    _kpi_card(mi4, "FPR (오탐율)", f"{model_row['FPR_Mean']*100:.2f}%", key="accent-model-fpr")
                    _notice(
                        f"※ {model_row['Candidate']} — 최소 Recall {model_row['Recall_Min']*100:.1f}%"
                        f"(불량 9건 중 최소 7건 검출). {model_row['Dashboard_Usage']}에 쓰입니다.",
                        kind="info",
                    )

                # 파생변수 표는 "모델" 얘기 쪽에 두기로 함(요청 반영, 2026-09) — 모델 판정
                # 바로 아래에서 "이 값들 때문에 이 판정" 흐름이 자연스럽고, 아래 인사이트·SHAP도
                # 이 표를 그대로 근거로 쓴다.
                lot_df = sd.get_lot_subset(df, lot_date, lot_slot)
                var_table = sd.get_variable_table(lot_df)
                var_table_shown = var_table[var_table["변수"].map(show_vars).fillna(True)]
                st.markdown(f"**주요변수 파생변수 — {lot_label} (모델 학습 사용 지표)**")
                _render_variable_table(var_table_shown)
                _render_deviation_rates(var_table_shown)

                # 인사이트: 기존엔 이탈률 기준 규칙기반 문장이었는데, SHAP이 이탈률을
                # 포함한 12개 피처 전부를 이미 반영하고 실제 모델 근거와 더 일치해서
                # 규칙기반은 빼고 SHAP 하나로 통합함(요청 반영, 2026-09) — SHAP 차트 →
                # SHAP 기반 인사이트 문장 → AI 자동 리포트 순서.
                st.subheader(
                    "인사이트",
                    help=(
                        "채택 모델(XGBoost 12F)이 이 LOT을 판정할 때 실제로 어떤 피처를 "
                        "얼마나 근거로 삼았는지(SHAP 기여도)를 보여주고, 그 결과를 문장으로 "
                        "풀어서 설명합니다."
                    ),
                )
                shap_contrib = None
                try:
                    shap_contrib = md.get_shap_contributions(lot_df)
                    st.altair_chart(ch.shap_contribution_chart(shap_contrib), width="stretch")
                    st.caption("빨강 = 불량 쪽으로 민 피처 · 파랑 = 정상 쪽으로 민 피처 (절댓값 큰 상위 6개)")
                    _notice(_shap_insight(shap_contrib), kind="info")
                except Exception:
                    _notice(
                        "SHAP 인사이트를 계산할 모델 파일을 찾을 수 없어 이 섹션은 건너뜁니다.",
                        kind="warn",
                    )

                # AI 자동 리포트(OpenAI, gpt-4o-mini) — 4가지: 1.전체 요약 2.관리도
                # 요약 3.모델 요약 4.SHAP 요약(기존 기능). 개별 생성 버튼은 없애고
                # "AI 자동 리포트 생성" 버튼 하나로 4개를 한 번에 채움(요청 반영,
                # 2026-09) — 하나가 실패해도 나머지는 정상 표시된다.

                # 1. 전체 요약용 데이터 — 이 탭엔 "데이터" 탭의 KPI 계산이 없어서 여기서
                # 다시 구함("로트 분석" 탭이 lot_ts 등을 독립적으로 다시 구하는 것과 같은 패턴).
                ov_start_d, ov_end_d = st.session_state.get(
                    "period", (df["date"].min(), df["date"].max())
                )
                ov_period_df = df[(df["date"] >= ov_start_d) & (df["date"] <= ov_end_d)]
                ov_kpi = sd.get_overview_kpi(ov_period_df)
                ov_ooc = md.get_out_of_control_rate(timeseries, ov_start_d, ov_end_d)
                ov_trend = md.get_recent_defect_rate_trend(df, ov_end_d)
                ov_period_label = f"{ov_start_d} ~ {ov_end_d}"

                # 2. 관리도 요약용 데이터 — 선택 LOT 건수 "앞에" 전체 기간 건수도 같이
                # 줘야 한다는 요청 반영(2026-09): 전체 기간 넬슨룰 위반 건수(rate뿐 아니라
                # rule1/rule5 건수까지) + 선택 LOT 건수, 프롬프트에서도 전체→LOT 순서로 배치.
                period_nelson = md.get_period_nelson_violations(timeseries, ov_start_d, ov_end_d)
                lot_nelson = md.get_lot_nelson_violations(lot_ts)

                report_specs = [
                    (
                        "전체 요약",
                        f"ai_overview_{ov_start_d}_{ov_end_d}",
                        lambda: ai_report.generate_overview_report(
                            ov_period_label, ov_kpi, ov_ooc, ov_trend
                        ),
                    ),
                    (
                        "관리도 요약 리포트",
                        f"ai_control_{ov_start_d}_{ov_end_d}_{lot_date}_{lot_slot}",
                        lambda: ai_report.generate_control_chart_report(
                            lot_label, ov_period_label, period_nelson, lot_nelson
                        ),
                    ),
                    (
                        "모델 요약",
                        "ai_model_summary",
                        lambda: ai_report.generate_model_report(model_row),
                    ),
                ]
                if shap_contrib is not None and final_oof:
                    report_specs.append((
                        "SHAP 요약",
                        f"ai_shap_{lot_date}_{lot_slot}",
                        lambda: ai_report.generate_shap_report(
                            lot_label, pred, final_oof["actual_label"], shap_contrib
                        ),
                    ))

                st.markdown("**AI 자동 리포트**")
                if st.button("🤖 AI 자동 리포트 생성", key="ai_report_generate_all"):
                    with st.spinner("리포트 생성 중..."):
                        for r_label, r_key, r_fn in report_specs:
                            try:
                                st.session_state[r_key] = r_fn()
                            except Exception as e:
                                st.session_state[r_key] = None
                                _notice(f"{r_label} 생성에 실패했습니다: {e}", kind="warn")

                st.markdown("1. 전체 요약")
                _ai_report_section(report_specs[0][0], report_specs[0][1])

                st.markdown("2. 관리도 요약")
                _ai_report_section(report_specs[1][0], report_specs[1][1])

                st.markdown("3. 모델 요약")
                _ai_report_section(report_specs[2][0], report_specs[2][1])

                st.markdown("4. SHAP 요약")
                if len(report_specs) > 3:
                    _ai_report_section(report_specs[3][0], report_specs[3][1])
                else:
                    st.caption("🤖 SHAP 요약 — SHAP 결과·모델 판정이 없어 생성할 수 없습니다.")
            else:
                st.warning("해당 LOT의 시계열 데이터가 없습니다.")
        else:
            if lot_label == "전체":
                st.info("특정 LOT을 선택하면 모델 판정·인사이트를 볼 수 있습니다.")
            else:
                st.warning("사이드바에서 LOT을 선택해주세요.")
else:
    meta = imgd.MODEL_META
    total_imgs = meta["normal_count"] + meta["defect_count"]
    defect_rate = meta["defect_count"] / total_imgs * 100

    st.subheader("검사 현황")
    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "누적 검사 개수", f"{total_imgs:,}장", key="kpi-total-imgs")
    _kpi_card(c2, "정상 데이터 수", f"{meta['normal_count']:,}장", key="kpi-normal-imgs")
    _kpi_card(c3, "불량 데이터 수", f"{meta['defect_count']}장", key="kpi-defect-imgs")
    _kpi_card(c4, "누적 불량률", f"{defect_rate:.1f}%", key="accent-defect-rate")

    # AI 요약/AI 보고서는 이 페이지엔 필요 없다는 판단으로 제거(요청 반영, 2026-09).
    # 대신 예전에 화면에서 뺐던 "5-seed 반복 검증 모델 비교" 표를 부활시킴 —
    # image_data.MODEL_COMPARISON 데이터 자체는 계속 남아있었고 화면에만 안 쓰였음.
    st.markdown("**모델 비교 (5-seed 반복 검증)**")
    st.table(imgd.MODEL_COMPARISON.set_index("모델"))
    st.caption(
        f"최종 선정: **{meta['final_model']}** — F1 {meta['f1_mean']:.3f} ± {meta['f1_std']:.3f}. "
        f"불량 {meta['defect_count']}장은 근접중복 제거 후 실질 {meta['unique_defect_groups']}개 "
        f"결함 그룹으로 확인됨."
    )

    # "판정 확률 분포" 섹션은 팀 피드백으로 삭제(2026-09) — 필요 없다는 판단.
    # imgd.load_confidence_scores / ch.confidence_distribution_chart 함수 자체는 남겨둠.

    # 순서 변경(튜터 피드백): 갤러리를 "선택 이미지 조회"보다 위로 배치.
    with st.expander("전체 이미지 갤러리", expanded=True):
        gallery_search = st.text_input(
            "검색",
            placeholder="번호나 이름으로 검색 (예: 3 → 3번 이미지, error → 불량 전체)",
            label_visibility="collapsed",
        )

        gal_col1, gal_col2, gal_col3, gal_col4 = st.columns([1, 1, 1, 1])
        with gal_col1:
            gallery_filter = st.radio(
                "필터", ["전체", "정상", "불량"], horizontal=True, label_visibility="collapsed"
            )
        with gal_col2:
            gallery_sort = st.radio(
                "정렬", ["과거순", "최신순"], horizontal=True, label_visibility="collapsed"
            )
        with gal_col3:
            gallery_view = st.radio(
                "보기 방식", ["썸네일 보기", "목록 보기"], horizontal=True, label_visibility="collapsed"
            )
        with gal_col4:
            if st.button("🔄 새로고침"):
                count = imgd.refresh_samples()
                st.session_state.gallery_page = 1
                st.session_state.gallery_refresh_msg = f"✅ 새로고침 완료 — 총 {count:,}장"
            if st.session_state.get("gallery_refresh_msg"):
                st.caption(st.session_state.gallery_refresh_msg)

        if gallery_filter == "전체":
            filtered = list(imgd.SAMPLE_IMAGES)
        else:
            filtered = [s for s in imgd.SAMPLE_IMAGES if s.category == gallery_filter]

        query = gallery_search.strip()
        if query:
            if query.isdigit():
                # 숫자만 입력하면 그 번호와 정확히 일치하는 이미지만
                # (부분일치로 하면 "3" 입력 시 13, 23, 30~39도 같이 걸려서 원하는 의도와 어긋남)
                target_n = int(query)
                filtered = [s for s in filtered if s.sort_key == target_n]
            else:
                q_lower = query.lower()
                filtered = [s for s in filtered if q_lower in s.label.lower()]

        filtered.sort(key=lambda s: s.sort_key, reverse=(gallery_sort == "최신순"))

        st.caption(f"총 {len(filtered):,}장")

        # 썸네일 보기: 열은 늘리고 행은 줄여서 한 페이지에 차지하는 공간을 줄임
        # (6열 × 2행 = 12장/페이지 → 자연히 페이지 수도 늘어남)
        THUMB_COLS = 6
        THUMB_ROWS = 2
        PAGE_SIZE = (THUMB_COLS * THUMB_ROWS) if gallery_view == "썸네일 보기" else 150
        total_pages = max(1, (len(filtered) - 1) // PAGE_SIZE + 1)

        if "gallery_page" not in st.session_state:
            st.session_state.gallery_page = 1
        st.session_state.gallery_page = min(st.session_state.gallery_page, total_pages)

        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("◀ 이전", disabled=st.session_state.gallery_page <= 1):
                st.session_state.gallery_page -= 1
                st.rerun()
        with nav_col2:
            st.markdown(
                f"<div style='text-align:center'>{total_pages}페이지 중 "
                f"{st.session_state.gallery_page}페이지</div>",
                unsafe_allow_html=True,
            )
        with nav_col3:
            if st.button("다음 ▶", disabled=st.session_state.gallery_page >= total_pages):
                st.session_state.gallery_page += 1
                st.rerun()

        page = st.session_state.gallery_page
        start_idx = (page - 1) * PAGE_SIZE
        page_items = filtered[start_idx:start_idx + PAGE_SIZE]

        if gallery_view == "썸네일 보기":
            gallery_cols = st.columns(THUMB_COLS)
            for i, s in enumerate(page_items):
                with gallery_cols[i % THUMB_COLS]:
                    st.image(str(s.original_path), width="stretch")
                    st.caption(f"{s.label} ({s.category})")
                    if st.button("조회", key=f"gallery_select_{s.label}"):
                        st.session_state.pending_img_category = s.category
                        st.session_state.pending_selected_img_label = s.label
                        st.rerun()
        else:
            # 폴더 탐색기처럼 이름·번호·대분류만 나열해서 고르는 목록 보기
            for s in page_items:
                row_col1, row_col2, row_col3 = st.columns([3, 2, 1])
                with row_col1:
                    st.write(f"📄 {s.label}")
                with row_col2:
                    st.write(s.category)
                with row_col3:
                    if st.button("조회", key=f"gallery_list_select_{s.label}"):
                        st.session_state.pending_img_category = s.category
                        st.session_state.pending_selected_img_label = s.label
                        st.rerun()

    st.subheader("선택 이미지 조회")
    sel_label = st.session_state.get("selected_img_label")
    sample = imgd.get_sample(sel_label) if sel_label else None
    if sample is None:
        st.warning("사이드바에서 조회할 이미지를 선택해주세요.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("원본")
            st.image(str(sample.original_path), width="stretch")
        with col2:
            if sample.gradcam_path and sample.gradcam_path.exists():
                st.caption("Grad-CAM")
                st.image(str(sample.gradcam_path), width="stretch")
            else:
                st.caption("Grad-CAM 없음")

    st.subheader("새 이미지 업로드 → 실시간 판정")
    uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded is not None:
        img_bytes = uploaded.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()[:8]

        with st.spinner("추론 및 Grad-CAM 생성 중..."):
            result = imgd.predict_image(img_bytes)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(uploaded, caption="업로드한 이미지", width="stretch")
            st.caption(f"디버그용 해시: {img_hash} (다른 이미지면 값이 달라야 함)")

        if result["available"]:
            with col_b:
                if result.get("gradcam_image"):
                    st.image(result["gradcam_image"], caption="Grad-CAM (판정 근거)", width="stretch")
                else:
                    st.caption("Grad-CAM 생성 실패")
            with col_c:
                badge = "🔴 불량" if result["prediction"] == "불량" else "🟢 정상"
                st.metric("자동판정", badge)
                st.metric("판정 확률", f"{result['confidence']*100:.4f}%")
                st.caption(
                    f"불량 확률 {result['probs']['불량']*100:.4f}% · "
                    f"정상 확률 {result['probs']['정상']*100:.4f}%"
                )
                st.caption("※ Grad-CAM은 AI가 최종 판정한 클래스를 기준으로 생성됩니다.")
        else:
            with col_b:
                st.warning(result["message"])
