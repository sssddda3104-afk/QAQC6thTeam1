import streamlit as st
import pandas as pd
import hashlib
import re
import sensor_data as sd
import model_data as md
import chart_helpers as ch
import image_data as imgd
import ai_report
import insight_analysis as ia
import dashboard_ui as ui

st.set_page_config(page_title="크로메이트 도금 공정 모니터링", layout="wide")

st.markdown(ui.CSS, unsafe_allow_html=True)


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
                  border-top-color:#2563eb; border-radius:50%;
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


def _on_heatmap_select() -> None:
    """Selection callbacks execute before sidebar widgets are instantiated."""
    event = st.session_state.get('lot_heatmap', {})
    points = event.get('selection', {}).get('lot_pick', [])
    if not points:
        return
    point = points[-1]
    try:
        date_str = str(point['date_key'])
        lot_num = int(point['lot'])
    except (KeyError, ValueError, TypeError):
        return
    valid = lots[(lots['date'].astype(str) == date_str) & (lots['lot'] == lot_num) & (lots['error'] == 1)]
    period = st.session_state.get('period')
    if valid.empty or (period and not period[0] <= valid.iloc[0]['date'] <= period[1]):
        return
    _goto_lot(date_str, lot_num)


def _notice(text: str, kind: str = "info") -> None:
    """참고한 타사 대시보드(TFinder)처럼, 중요 안내문을 파스텔 컬러 박스로 강조.
    기본 st.info()/st.caption()보다 눈에 잘 들어오고, 색으로 "좋은 소식(info)"과
    "주의/미채택 등(warn)"을 구분해서 전달한다(2026-09, 디자인 참고 반영).
    kind: "info"(연초록, 설명·공지용) 또는 "warn"(연노랑, 주의·제약사항용)."""
    palette = {
        "info": ("#edf4ff", "#2457a7", "#ccddf7"),
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
    """Explain the strongest contributions without presenting them as causes."""
    from html import escape
    if contrib.empty:
        return "표시할 SHAP 결과가 없습니다."
    ranked = contrib.loc[contrib['shap'].abs().sort_values(ascending=False).index]
    lines = ['<b>판정 근거 핵심 해석</b><br>아래 항목은 영향의 절댓값이 큰 순서입니다. 빨강(+)은 불량 방향, 파랑(−)은 정상 방향으로 모델 점수를 움직인 근거입니다.']
    for i, row in ranked.head(top_n).reset_index(drop=True).iterrows():
        label = str(row['label'])
        direction = '불량 방향' if row['shap'] > 0 else ('정상 방향' if row['shap'] < 0 else '영향 없음')
        if '표준편차' in label:
            meaning = '공정 중 측정값의 흔들림 정도입니다. 값의 수준뿐 아니라 순간 변동과 제어 기록을 확인하세요.'
        elif 'IQR' in label:
            meaning = '측정값 가운데 50%가 차지하는 범위입니다. 일부 극단값보다 전반적인 변동 폭을 나타내므로 관리도의 분포와 함께 확인하세요.'
        elif '최솟값' in label or '최소' in label:
            meaning = 'LOT에서 기록된 가장 낮은 측정값입니다. 해당 시점의 지속 여부와 센서·작업 기록을 대조하세요.'
        elif '이탈' in label:
            meaning = '정해진 관리 기준을 벗어난 측정값의 비율입니다. 어느 시점과 방향에서 이탈했는지 관리도를 확인하세요.'
        else:
            meaning = '이 항목의 측정 기록과 관리도에서 일시적 변화인지 반복되는 패턴인지 확인하세요.'
        lines.append(f'<p><b>{i+1}. {escape(label)}</b> · 값 <b>{row["value"]:.3g}</b><br>모델 영향: <b>{direction}</b> (SHAP {row["shap"]:+.3f}). {meaning}</p>')
    lines.append('정상 방향의 기여가 있어도 최종 판정은 불량일 수 있습니다. 모델은 모든 항목의 기여를 함께 반영합니다. <b>SHAP 크기는 불량 확률이나 조정 목표값이 아니며, 원인·조정 효과를 증명하지 않습니다.</b>')
    return ''.join(lines)


def _kpi_card(col, label: str, value: str, key: str, help: str | None = None) -> None:
    """Render owned HTML markup so KPI styling does not depend on widget wrappers."""
    with col.container(key=key):
        st.markdown(ui.kpi_html(label, value, help), unsafe_allow_html=True)


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
    # Explicit table layout prevents column widths from expanding with the page.
    styled = styled.set_table_attributes('style="width:100%;table-layout:fixed;border-collapse:collapse"')
    table_html = styled.hide(axis="index", names=True).to_html()
    table_html = re.sub(
        r'(<thead>\s*<tr>\s*<th\b[^>]*>).*?(</th>)',
        r'\1변수\2', table_html, count=1, flags=re.S,
    )
    st.html(
        '<style>' 
        '.lot-stats-table{width:100%;max-width:640px;box-sizing:border-box;}'
        '.lot-stats-table table{font-size:14px;color:#172b4d;}'
        '.lot-stats-table th,.lot-stats-table td{padding:10px 12px;border:1px solid #dce3ed;box-sizing:border-box;overflow-wrap:anywhere;}'
        '.lot-stats-table th{text-align:left;}'
        '.lot-stats-table td{text-align:right;font-variant-numeric:tabular-nums;}'
        '.lot-stats-table tr>*:first-child{width:22%;}'
        '.lot-stats-table tr>*:not(:first-child){width:26%;}'
        '</style><div class="lot-stats-table">' + table_html + '</div>'
    )


def _render_deviation_rates(var_table: pd.DataFrame) -> None:
    """이탈률(%)만 변수별 진행바로 따로 표시. LOT 하나(그 자체의 시계열) 안에서
    측정값이 기준을 벗어난 비율이라는 걸 캡션으로 같이 안내한다."""
    st.caption("이탈률 — 이 LOT 자체 측정값 중 기준을 벗어난 비율")
    for _, row in var_table.iterrows():
        c1, c2 = st.columns([1, 1.4], gap="small")
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
        "cc_show_sigma3": True,
        "cc_show_iqr": False, "cc_show_min": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    with st.expander("관리도 표시 옵션", expanded=False):
        o1, o2, o3 = st.columns(3)
        o1.checkbox("±3σ 기준선 (점선)", key="cc_show_sigma3")
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
# 이전 로트 분석 탭을 보고 있던 세션도 통합 화면으로 이동한다.
if st.session_state.get("sensor_tab") not in ("데이터", "모델·인사이트"):
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
    # 조회와 진행 분석은 데이터 화면에서 연결한다.
    st.segmented_control(
        "탭 선택",
        ["데이터", "모델·인사이트"],
        label_visibility="collapsed",
        key="sensor_tab",
        required=True,
    )

    if st.session_state.sensor_tab == "데이터":
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
        st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
        heat_col, list_col = st.columns([2, 1], gap="large")
        with heat_col:
            st.markdown("**전체 LOT 중 불량 위치**")
            with st.container(border=True, key="heatmap-panel"):
                st.altair_chart(
                    ch.defect_heatmap_chart(sd.get_lot_list(period_df), st.session_state.get("selected_lot")),
                    width="stretch", key="lot_heatmap",
                    on_select=_on_heatmap_select, selection_mode=["lot_pick"]
                )
            st.caption("회색 = 정상, 빨강 = 불량 (정상 칸은 마우스를 올려도 반응하지 않습니다)")
        with list_col:
            st.markdown("**불량 발생 목록**")
            defect_rows = sd.get_lot_list(period_df)
            defect_rows = defect_rows[defect_rows["error"] == 1].sort_values("date")
            with st.container(height=420, key="defect-list-box"):
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

        st.caption("히트맵의 빨간 불량 칸을 클릭하면 아래 상세 조회가 갱신됩니다. 빨강 = 불량")

        st.divider()
        sel = st.session_state.get("selected_lot")
        lot_label = st.session_state.get("selected_lot_label")
        if not sel:
            if lot_label == "전체":
                st.info("히트맵의 불량 칸이나 사이드바에서 LOT을 선택하면 이곳에 상세 분석이 표시됩니다.")
            else:
                st.warning("사이드바에서 LOT을 선택해주세요.")
        else:
            lot_date, lot_slot = sel

            st.subheader(f"LOT 상세 분석 — {lot_label}")
            progress = st.slider("진행률", min_value=0, max_value=100, value=100, step=1, key="lot_progress")

            lot_ts = md.get_lot_timeseries(timeseries, lot_date, lot_slot)
            total_points = len(lot_ts)

            if total_points == 0:
                st.warning("해당 LOT의 시계열 데이터가 없습니다.")
            else:
                # 그래프·표는 실제 진행률(1% 단위)에 맞춰 즉시 계산
                upto = max(1, round(total_points * progress / 100))
                current = lot_ts.iloc[upto - 1]
                available = [p for p in md.PROGRESS_LEVELS if p <= progress]
                checkpoint = max(available) if available else None
                ew = md.get_ew_oof(progress_cp, lot_date, lot_slot, checkpoint) if checkpoint is not None else None
                risk = md.risk_status_label(ew["risk_percentile"]) if ew else "데이터 부족"
                alert = ("예" if ew["threshold_alert"] else "아니오") if ew else "—"
                basis = f"최근 체크포인트 {checkpoint}% 기준입니다." if checkpoint is not None else "아직 사용 가능한 체크포인트가 없습니다."
                risk_help = "최종 채택 모델이 아닌 참고용 Early Warning 신호입니다. 확정 불량 판정에는 사용하지 않습니다.\n\n" + basis
                alert_help = "해당 체크포인트의 Early Warning 확률이 경보 기준값을 초과했는지 표시합니다. '예'는 확정 불량 판정이 아닙니다.\n\n" + basis
                is_defect = bool(lot_ts["Defect"].eq(1).any())
                st.markdown(ui.process_html(progress, current, risk, alert, risk_help, alert_help, is_defect=is_defect), unsafe_allow_html=True)
                if ew and checkpoint != progress:
                    st.caption(f"AI 참고 위험도·경보 여부는 {checkpoint}% 체크포인트 기준입니다.")

                show_vars = st.session_state.get("show_vars", {"pH": True, "온도": True, "전압": True})

                st.subheader("관리도")
                cc_opts = _control_chart_options()
                st.caption("빨간 점선 = 평균 ±3σ · 회색 점선 = 평균")
                for var_name in ["pH", "온도", "전압"]:
                    if show_vars.get(var_name, True):
                        with st.container(border=True, key=f"control-panel-{var_name}"):
                            st.altair_chart(
                                ch.variable_control_chart(
                                    lot_ts, var_name, ref_stats=ref_stats[var_name], upto=upto,
                                    **cc_opts
                                ),
                                width="stretch",
                            )

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
                    st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
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
                st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
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
                st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
                st.markdown(f"**주요변수 파생변수 — {lot_label} (모델 학습 사용 지표)**")
                table_col, rates_col = st.columns([1.15, 1], gap="large")
                with table_col:
                    st.caption("주요변수 통계")
                    _render_variable_table(var_table_shown)
                with rates_col:
                    _render_deviation_rates(var_table_shown)

                # 인사이트: 기존엔 이탈률 기준 규칙기반 문장이었는데, SHAP이 이탈률을
                # 포함한 12개 피처 전부를 이미 반영하고 실제 모델 근거와 더 일치해서
                # 규칙기반은 빼고 SHAP 하나로 통합함(요청 반영, 2026-09) — SHAP 차트 →
                # SHAP 기반 인사이트 문장 → AI 자동 리포트 순서.
                st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
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
                    cross = ia.control_crosscheck(shap_contrib, lot_ts, ref_stats)
                    st.markdown("**핵심 요약**")
                    st.markdown(ia.direction_summary(shap_contrib))
                    st.altair_chart(ch.shap_contribution_chart(shap_contrib), width="stretch")
                    st.caption("빨강 = 불량 방향 · 파랑 = 정상 방향 · 막대가 길수록 모델 점수에 미친 영향이 큽니다.")
                    st.markdown("**이렇게 해석할 수 있습니다**")
                    for explanation in ia.evidence_explanations(shap_contrib, cross):
                        st.markdown(explanation)
                    st.caption("모델의 판단 근거를 설명하는 것이며, 불량 원인이나 조정 효과를 확정하는 분석은 아닙니다.")
                    with st.expander("상세 수치 · 계산 기준", expanded=False):
                        st.caption("SHAP 이탈률은 모델 학습 설정의 상한 초과 비율이며 %로 환산했습니다. 화면의 팀 관리이탈 기준과 다를 수 있습니다.")
                        st.markdown("**관리도 교차 확인**")
                        st.dataframe(cross, hide_index=True, width="stretch")
                        st.caption("선택 LOT 전체 구간을 관리도와 동일한 평균·표준편차로 계산합니다. 조기 경고에서는 ±3σ 초과점을 제외했습니다. 변수별 모델 근거는 절댓값이 가장 큰 피처입니다.")
                        st.markdown("**전체 SHAP 수치**")
                        details = shap_contrib[["label", "value", "shap"]].copy()
                        details = details.rename(columns={"label":"판정 근거 항목", "value":"피처 값(모델 입력 단위)", "shap":"SHAP 기여도"})
                        st.dataframe(details, hide_index=True, width="stretch")
                        st.caption("전체 피처를 표시하며, 기여도는 백분율이 아닙니다. SHAP 영향 순위는 점검 우선순위와 다릅니다.")
                        st.caption("SHAP은 배포된 XGBoost 모델의 설명입니다. 상단 모델 판정은 별도 OOF 검증 결과이므로 해당 OOF 판정의 직접 설명은 아닙니다.")
                except Exception:
                    _notice(
                        "SHAP 인사이트를 계산할 모델 파일을 찾을 수 없어 이 섹션은 건너뜁니다.",
                        kind="warn",
                    )


                lot_nelson = md.get_lot_nelson_violations(lot_ts)
                st.markdown('<div class="section-gap" aria-hidden="true"></div>', unsafe_allow_html=True)
                st.subheader("AI 자동 리포트")
                report_end = st.session_state.get("period", (None, df["date"].max()))[1]
                report_start = pd.Timestamp(report_end).date() - pd.Timedelta(days=6)
                st.caption(f"최근 7일: {report_start} ~ {report_end} · 선택 LOT: {lot_label}")
                report_key = f"ai_operator_v6_{report_end}_{lot_date}_{lot_slot}"
                if st.button("AI 자동 리포트", key="ai_operator_generate"):
                    with st.spinner("AI 자동 리포트 작성 중..."):
                        try:
                            context = ai_report.build_operator_context(timeseries, report_end, lot_ts, ref_stats)
                            recent_nelson = md.get_period_nelson_violations(timeseries, pd.Timestamp(report_start).date(), report_end)
                            st.session_state[report_key] = ai_report.generate_operator_report(lot_label, lot_nelson, final_oof, context, recent_nelson, shap_contrib)
                        except Exception:
                            st.error("보고서를 생성하지 못했습니다. API 키와 연결 상태를 확인한 뒤 다시 시도해주세요.")
                if st.session_state.get(report_key):
                    st.markdown(st.session_state[report_key])
                else:
                    st.caption("이번 LOT에서 반복된 문제와 먼저 확인할 기록을 쉬운 말로 정리합니다. 최근 7일 현황은 참고 근거로 함께 제공합니다.")
                st.caption("AI 보고서는 점검 참고용입니다. 실제 조치는 현장 절차와 담당자 확인에 따릅니다.")
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
