"""SHAP explanations and control-chart signals for the selected LOT."""
import pandas as pd
import chart_helpers as ch

def control_crosscheck(contrib, lot_ts, ref_stats):
    ranked = contrib.sort_values("shap",key=lambda x:x.abs(),ascending=False)
    cross = []
    for label,col in [("pH","pH"),("온도","Temp"),("전압","Voltage")]:
        subset = ranked[ranked.feature.str.startswith(col+"_")]
        if subset.empty: continue
        strongest = subset.iloc[0]
        stats = ref_stats[label]
        std = stats["std"]
        if pd.isna(std) or std <= 0:
            n1,n5 = None,None
        else:
            z = (lot_ts[col]-stats["mean"])/std
            rule1,rule5=ch._nelson_rule1(z),ch._nelson_rule5(z)
            n1,n5=int(rule1.sum()),int((rule5 & ~rule1).sum())
        direction = "불량 방향" if strongest.shap>0 else ("정상 방향" if strongest.shap<0 else "영향 없음")
        if n1 is None:
            interpretation="기준 통계 부족으로 교차 확인 불가"
        elif n1+n5:
            interpretation="통계 신호와 불량 방향 근거가 함께 관찰됨" if strongest.shap>0 else "통계 신호가 있으나 주요 모델 근거는 정상 방향" if strongest.shap<0 else "통계 신호가 있으나 해당 피처의 모델 영향은 없음"
        else:
            interpretation="통계 신호 없이 불량 방향 근거가 관찰됨" if strongest.shap>0 else "해당 규칙 신호 없음; 주요 근거는 "+direction
        cross.append({"변수":label,"가장 큰 모델 근거":f"{strongest['label']} ({strongest.shap:+.2f})",
            "±3σ 초과":"확인 불가" if n1 is None else f"{n1}개",
            "조기 경고":"확인 불가" if n5 is None else f"{n5}개","교차 해석":interpretation})
    return pd.DataFrame(cross)


def direction_summary(contrib):
    parts=[]
    for positive,label in [(True,"불량"),(False,"정상")]:
        group=contrib[contrib.shap>0] if positive else contrib[contrib.shap<0]
        if not group.empty:
            row=group.loc[group.shap.abs().idxmax()]
            parts.append(f"**{row['label']}**이 {label} 방향으로 가장 크게 작용했습니다.")
    return " ".join(parts) if parts else "이번 LOT에서 확인된 SHAP 기여도는 모두 0입니다."


def evidence_explanations(contrib, cross):
    """Three short explanations of selected values, model direction and signals."""
    ranked=contrib.sort_values('shap',key=lambda x:x.abs(),ascending=False).head(3)
    explanations=[]
    for i,(_,feature) in enumerate(ranked.iterrows()):
        prefix=feature['feature'].split('_')[0]
        var={'pH':'pH','Temp':'온도','Voltage':'전압'}.get(prefix,prefix)
        signal=cross[cross['변수']==var]
        direction='불량' if feature['shap']>0 else '정상' if feature['shap']<0 else None
        role=f"모델 점수를 **{direction} 방향**으로 움직였습니다." if direction else "모델 점수에 영향을 주지 않았습니다."
        rate = feature['feature'].endswith('_rate')
        value = float(feature['value']) * (100 if rate else 1)
        unit = '%' if rate else (' °C' if prefix == 'Temp' else (' V' if prefix == 'Voltage' else ''))
        text=(f"**{i+1}. {feature['label']}** — {role}\n\n"
              f"선택 LOT의 값은 **{value:.2f}{unit}**입니다.")
        if not signal.empty:
            row=signal.iloc[0]
            if row['±3σ 초과']=='확인 불가':
                text+=f" {var} 관리도는 기준 통계가 부족해 비교하지 못했습니다."
            elif row['±3σ 초과']=='0개' and row['조기 경고']=='0개':
                text+=f" {var} 관리도에서는 ±3σ 초과와 조기 경고가 없었습니다. 모델이 중요하게 본 특성이 반드시 관리도 이상으로 나타나는 것은 아닙니다."
            else:
                text+=f" {var} 관리도에는 **±3σ 초과 {row['±3σ 초과']}·조기 경고 {row['조기 경고']}**가 있습니다. 이는 같은 변수에서 관찰된 별도의 통계 신호입니다."
        explanations.append(text)
    return explanations
