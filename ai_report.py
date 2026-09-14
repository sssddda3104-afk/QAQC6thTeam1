"""AI 자동 리포트 — OpenAI API 연동(2026-09).

"인사이트" 섹션의 "AI 자동 리포트"는 4가지를 따로 생성한다(요청 반영):
1. 전체 요약 — 선택 기간 전체 정상/불량 데이터 현황
2. 관리도 요약 — 선택 LOT의 넬슨룰 위반 건수 + 선택 기간 전체 관리이탈율
3. 모델 요약 — 채택 모델(XGBoost 12F)의 성능 검증 지표
4. SHAP 요약 — 선택 LOT의 SHAP 기여도 기반 요약(기존 기능)

4개 모두 버튼 클릭 시에만 호출하고(비용·지연 방지), app.py에서 각자의 키로
session_state에 캐싱해서 같은 조회 기간/LOT을 다시 봐도 재호출하지 않는다.

API 키는 st.secrets["OPENAI_API_KEY"]를 우선 쓰고, 없으면 환경변수
OPENAI_API_KEY를 쓴다(로컬 실행 편의) — 둘 다 없으면 안내 메시지가 담긴
RuntimeError를 던진다. openai 패키지는 함수 안에서 지연 import한다 — 패키지가
아직 안 깔려 있어도 대시보드의 나머지 기능은 정상 동작하게 하기 위함.
"""
import os

import pandas as pd
import streamlit as st

DEFAULT_MODEL = "gpt-4o-mini"

# 4개 리포트 전부에 공통으로 지키는 규칙. 특히 2번(인과관계 단정 금지)은 원래
# SHAP 리포트에만 있었는데, 관리도·전체 요약·모델 요약도 "그래서 이렇게 하면
# 좋아진다"는 근거 없는 인과 조언을 만들어낼 수 있어 4개 전부로 확장함(2026-09).
_BASE_RULES = (
    "다음 규칙을 반드시 지키세요:\n"
    "1. 숫자를 지어내지 말고 입력으로 주어진 값만 그대로 인용하세요.\n"
    "2. 지표들 사이의 인과관계를 단정하지 마세요 — 상관관계·통계적 관찰 수준에서만 "
    "서술하고, \"이렇게 하면 해결됩니다\" 같은 확정적 원인·해결책 제시는 피하세요.\n"
    "3. 한국어로, 현장 엔지니어가 바로 읽을 수 있게 간결한 문단 2~3개로 작성하세요.\n"
    "4. 확정 진단이 아니라 참고용 리포트임을 마지막에 한 줄로 명시하세요."
)

_ROLE_PREFIX = "당신은 금속 크로메이트 도금 공정의 품질관리 엔지니어를 돕는 보고서 작성 보조입니다.\n"

SYSTEM_PROMPT_OVERVIEW = (
    _ROLE_PREFIX
    + "입력으로는 선택된 조회 기간의 전체 LOT 현황(총 LOT 수, 불량 LOT 수, 불량률, "
    "pH/온도/전압 평균, 변수별 관리이탈율, 최근 불량률 추이)이 주어집니다. 이를 "
    "바탕으로 정상/불량 데이터 전체 현황을 요약하는 리포트를 작성하세요.\n\n"
    + _BASE_RULES
)

SYSTEM_PROMPT_CONTROL = (
    _ROLE_PREFIX
    + "입력으로는 관리도(SPC) 기준 넬슨룰 위반 현황이 주어집니다 — 먼저 선택 기간 "
    "전체의 변수별 위반 건수(Rule 1: ±3시그마 초과, Rule 5: 조기 경고)와 관리이탈율(%), "
    "그다음 선택 LOT 하나의 위반 건수입니다. 전체 기간 대비 이 LOT이 어떤 수준인지도 "
    "함께 짚어 공정 안정성 관점의 리포트를 작성하세요.\n\n"
    + _BASE_RULES
)

SYSTEM_PROMPT_MODEL = (
    _ROLE_PREFIX
    + "입력으로는 채택된 불량 판정 모델(XGBoost 12F)의 성능 검증 지표(PR-AUC, "
    "Recall, FPR 등)가 주어집니다. 이를 바탕으로 이 모델을 얼마나 신뢰하고 "
    "어떻게 활용하면 좋을지 요약하는 리포트를 작성하세요.\n\n"
    + _BASE_RULES
)

# SHAP 리포트만 추가 규칙이 하나 더 있다 — "SHAP 기여도는 상관관계일 뿐"이라는
# 점을 규칙 2번보다 더 구체적으로(값을 낮추면/높이면 정상화된다는 표현 금지)
# 짚어줄 필요가 있어서(2026-09, 기존 프롬프트 유지).
SYSTEM_PROMPT_SHAP = (
    _ROLE_PREFIX
    + "입력으로는 한 LOT에 대한 채택 모델(XGBoost 12F)의 판정 결과와, 그 판정에 각 "
    "피처가 얼마나 기여했는지를 나타내는 SHAP 값이 주어집니다.\n\n"
    + _BASE_RULES
    + "\n5. SHAP 기여도는 상관관계 기반 수치일 뿐입니다. \"이 값을 낮추면/높이면 "
    "정상화됩니다\" 같은 표현 대신 \"~가 판정에 크게 작용했으니 현장에서 추가로 "
    "확인해볼 필요가 있습니다\"처럼 완곡하게 표현하세요."
)


def _get_api_key() -> str:
    """st.secrets를 우선 확인하고, 없으면 환경변수를 확인한다. secrets.toml 파일 자체가
    없는 로컬 환경에서는 st.secrets 접근이 예외를 던질 수 있어 try로 감싼다."""
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. .streamlit/secrets.toml "
            "(OPENAI_API_KEY = \"...\") 또는 환경변수로 설정해주세요."
        )
    return key


def _call_openai(
    system_prompt: str, user_prompt: str,
    model: str = DEFAULT_MODEL, max_tokens: int = 500,
) -> str:
    """4개 리포트 함수가 공통으로 쓰는 OpenAI 호출부. 실패(키 없음/네트워크/API
    오류)는 예외를 그대로 올려서 호출부(app.py)가 각 섹션의 try/except +
    _notice(warn) 패턴으로 처리하게 한다."""
    from openai import OpenAI  # 지연 import — openai 패키지가 없어도 나머지 기능엔 영향 없음

    client = OpenAI(api_key=_get_api_key())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ── 1. 전체 요약 ──────────────────────────────────────────────
def build_overview_prompt(period_label: str, kpi: dict, ooc: dict, trend: dict) -> str:
    return (
        f"조회 기간: {period_label}\n"
        f"총 LOT 수: {kpi['총 LOT 수']}건\n"
        f"불량 LOT 수: {kpi['불량 LOT 수']}건\n"
        f"불량률: {kpi['불량률(%)']}%\n"
        f"평균 pH: {kpi['평균 pH']}, 평균 온도: {kpi['평균 온도']}℃, "
        f"평균 전압: {kpi['평균 전압']}V\n"
        f"pH 관리이탈율: {ooc.get('pH')}%, 온도 관리이탈율: {ooc.get('온도')}%, "
        f"전압 관리이탈율: {ooc.get('전압')}%\n"
        f"최근 7일 불량률: {trend.get('현재')}% "
        f"(직전 7일 대비 변화: {trend.get('변화')})"
    )


def generate_overview_report(period_label: str, kpi: dict, ooc: dict, trend: dict, model: str = DEFAULT_MODEL) -> str:
    prompt = build_overview_prompt(period_label, kpi, ooc, trend)
    return _call_openai(SYSTEM_PROMPT_OVERVIEW, prompt, model=model)


# ── 2. 관리도 요약 ────────────────────────────────────────────
def build_control_chart_prompt(lot_label: str, period_label: str, period_nelson: dict, lot_nelson: dict) -> str:
    lines = [f"조회 기간: {period_label}", "", "이 기간 전체의 변수별 넬슨룰 위반 현황:"]
    for var, v in period_nelson.items():
        lines.append(
            f"- {var}: Rule 1(±3σ 초과) {v['rule1_count']}건, "
            f"Rule 5(조기 경고) {v['rule5_count']}건, 관리이탈율 {v['rate']}% "
            f"(총 {v['total_points']}개 측정점)"
        )
    lines.append("")
    lines.append(f"선택 LOT: {lot_label}")
    lines.append("이 LOT의 넬슨룰 위반 건수:")
    for var, v in lot_nelson.items():
        lines.append(
            f"- {var}: Rule 1(±3σ 초과) {v['rule1_count']}건, "
            f"Rule 5(조기 경고) {v['rule5_count']}건 (총 {v['total_points']}개 측정점)"
        )
    return "\n".join(lines)


def generate_control_chart_report(
    lot_label: str, period_label: str, period_nelson: dict, lot_nelson: dict, model: str = DEFAULT_MODEL,
) -> str:
    prompt = build_control_chart_prompt(lot_label, period_label, period_nelson, lot_nelson)
    return _call_openai(SYSTEM_PROMPT_CONTROL, prompt, model=model)


# ── 3. 모델 요약 ──────────────────────────────────────────────
def build_model_prompt(model_row: pd.Series) -> str:
    return (
        f"모델: {model_row['Candidate']}\n"
        f"PR-AUC: {model_row['PR_AUC_Mean']:.3f} (±{model_row['PR_AUC_SD']:.3f})\n"
        f"Recall 평균: {model_row['Recall_Mean']*100:.1f}% "
        f"(최소 {model_row['Recall_Min']*100:.1f}%)\n"
        f"FPR(오탐율): {model_row['FPR_Mean']*100:.2f}%\n"
        f"용도: {model_row['Dashboard_Usage']}"
    )


def generate_model_report(model_row: pd.Series, model: str = DEFAULT_MODEL) -> str:
    prompt = build_model_prompt(model_row)
    return _call_openai(SYSTEM_PROMPT_MODEL, prompt, model=model)


# ── 4. SHAP 요약 ──────────────────────────────────────────────
def build_shap_prompt(
    lot_label: str, pred: str, actual_label: str,
    shap_contrib: pd.DataFrame, top_n: int = 5,
) -> str:
    """SHAP 기여도 상위 top_n개 + LOT 메타를 모델 입력용 텍스트로 정리.
    파생변수 표 전체가 아니라 SHAP 상위 몇 개로 좁히는 이유: 토큰 비용을 줄이고,
    "모델이 실제로 근거로 삼은 것"에 집중하게 하기 위함(2026-09)."""
    rows = []
    for _, row in shap_contrib.head(top_n).iterrows():
        direction = "불량 쪽" if row["shap"] > 0 else "정상 쪽"
        rows.append(
            f"- {row['label']}: 값={row['value']:.3g}, "
            f"SHAP 기여도={row['shap']:+.3f} ({direction})"
        )
    shap_block = "\n".join(rows)
    return (
        f"LOT: {lot_label}\n"
        f"모델 판정: {pred}\n"
        f"실제 결과: {actual_label}\n\n"
        f"SHAP 기여도 상위 {top_n}개 (절댓값 큰 순):\n{shap_block}"
    )


def generate_shap_report(
    lot_label: str, pred: str, actual_label: str,
    shap_contrib: pd.DataFrame, model: str = DEFAULT_MODEL, top_n: int = 5,
) -> str:
    prompt = build_shap_prompt(lot_label, pred, actual_label, shap_contrib, top_n=top_n)
    return _call_openai(SYSTEM_PROMPT_SHAP, prompt, model=model)
