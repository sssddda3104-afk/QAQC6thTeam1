"""Presentation components independent of Streamlit's generated CSS classes."""
from html import escape

CSS = '''<style>
html, body, [data-testid="stApp"] {font-family:"Pretendard","Noto Sans KR","Malgun Gothic",sans-serif;color:#172b4d;}
[data-testid="stAppViewContainer"] {background:#f3f6fb;}
[data-testid="stHeader"] {background:#f3f6fb;}
section[data-testid="stSidebar"] {background:#eaf0f8;border-right:1px solid #dce5f1;}
h1 {font-size:2rem!important;font-weight:800!important;letter-spacing:-.055em;padding-bottom:1rem!important;}
h2 {font-size:1.45rem!important;letter-spacing:-.035em;}
h3 {font-size:1.15rem!important;letter-spacing:-.025em;}
[data-testid="stCaptionContainer"] {color:#526580;line-height:1.6;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
[data-testid="stSegmentedControl"] {background:#e5edf8;border-radius:12px;padding:6px;}
.blue-kpi {box-sizing:border-box;min-height:112px;background:#fff;border:1px solid #dce5f1;border-left:6px solid #2563eb;border-radius:14px;padding:19px 20px;box-shadow:0 4px 16px #16376608;}
.blue-kpi-label {display:flex;align-items:center;gap:7px;color:#34445c;font-size:14px;font-weight:650;line-height:1.45;}
.blue-kpi-value {color:#1d5bd7;font-size:clamp(23px,2.1vw,32px);font-weight:800;letter-spacing:-.045em;line-height:1.25;margin-top:5px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums;}
.blue-kpi-help {position:relative;cursor:help;color:#607695;font-size:12px;border:1px solid #b8c9e1;border-radius:50%;width:16px;height:16px;text-align:center;flex:none;}
.blue-kpi-help span {display:none;position:absolute;bottom:24px;left:-110px;width:240px;background:#172b4d;color:white;padding:12px;border-radius:8px;z-index:99;white-space:pre-line;font-weight:400;text-align:left;}
.blue-kpi-help:hover span,.blue-kpi-help:focus span {display:block;}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce5f1;border-left:4px solid #2563eb;border-radius:12px;padding:12px 16px;}
[data-testid="stMetricValue"] {color:#1d5bd7;font-size:1.7rem;}
.process-panel {background:white;border:1px solid #dce5f1;border-radius:18px;padding:20px;margin:8px 0 20px;}
.process-head {display:flex;justify-content:space-between;align-items:center;gap:16px;font-weight:700;color:#243b5b;}
.process-badge {background:#eaf1ff;color:#1d5bd7;padding:6px 12px;border-radius:30px;font-size:13px;}
.process-layout {display:grid;grid-template-columns:1.15fr 1fr;gap:28px;align-items:center;margin:18px 0;}
.process-layout svg {display:block;width:100%;max-height:390px;margin:0 auto;}
.process-metrics {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;}
@media(max-width:760px){.process-layout{grid-template-columns:1fr;gap:16px}.process-layout svg{max-height:280px}.process-metrics{gap:10px}}
.process-note {color:#60718a;font-size:12px;line-height:1.6;margin-top:10px;}
@media(max-width:640px){.blue-kpi{min-height:100px;padding:16px}.process-panel{padding:12px}h1{font-size:1.6rem!important}}

/* Consistent spacing and typography without changing application behavior. */
[data-testid="stMainBlockContainer"] {max-width:1600px;padding-top:2.2rem;padding-bottom:4rem;}
[data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#f6f8fc 0%,#f0f4f9 100%);}
section[data-testid="stSidebar"] {background:#edf2f9;}
h1 {font-size:1.85rem!important;letter-spacing:-.04em;line-height:1.35!important;}
h2 {font-size:1.35rem!important;line-height:1.45!important;padding-top:1.25rem!important;padding-bottom:.7rem!important;}
h3 {font-size:1.1rem!important;line-height:1.5!important;}
[data-testid="stCaptionContainer"] {font-size:.82rem;color:#617089;}
[data-testid="stSegmentedControl"] {padding:5px;background:#e8eef7;margin-bottom:12px;}
[data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"] {min-height:38px;border-radius:9px;}
[data-testid="stButton"] button {border-radius:9px;border-color:#d6e0ee;min-height:36px;font-weight:600;}
[data-testid="stButton"] button:hover {border-color:#2563eb;background:#eff5ff;color:#1d5bd7;}
[data-testid="stExpander"] {background:#ffffffb8;border-radius:12px;}
[data-testid="stVegaLiteChart"] {padding:0;border:0;min-width:0;}
.st-key-heatmap-panel, [class*="st-key-control-panel-"] {background:#fff;border-radius:14px;}
.section-gap {height:28px;}
.blue-kpi {min-height:116px;margin-bottom:8px;border-radius:13px;border-left-width:4px;padding:18px 20px;box-shadow:0 3px 12px #16376605;}
.blue-kpi-label {min-height:21px;align-items:flex-start;line-height:1.5;font-size:13px;letter-spacing:-.015em;}
.blue-kpi-label>span:first-child {min-width:0;overflow-wrap:break-word;}
.blue-kpi-value {font-size:clamp(24px,2vw,31px);line-height:1.3;margin-top:8px;letter-spacing:-.025em;}
.blue-kpi-help {margin-top:2px;line-height:16px;}
.blue-kpi-help span {line-height:1.65;box-shadow:0 6px 24px #172b4d26;font-size:12px;}
.blue-kpi-help:focus-visible {outline:2px solid #2563eb;outline-offset:3px;}
.process-panel {padding:24px 28px;margin:12px 0 24px;border-radius:18px;box-shadow:0 6px 24px #16376605;}
.process-head {font-size:16px;letter-spacing:-.02em;}
.process-badge {font-size:12px;white-space:nowrap;padding:7px 12px;}
.process-layout {grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:32px;margin:24px 0;}
.process-metrics {gap:12px;}
.process-metrics .blue-kpi {min-height:108px;margin:0;padding:16px 18px;background:#f8faff;box-shadow:none;}
.process-metrics .blue-kpi-value {font-size:27px;}
.process-note {margin-top:12px;font-variant-numeric:tabular-nums;}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;font-weight:700;}
@media(max-width:900px){.process-layout{grid-template-columns:1fr;gap:20px}.process-layout svg{max-height:270px}.process-panel{padding:20px}}
@media(max-width:640px){[data-testid="stMainBlockContainer"]{padding:1.2rem 1rem 3rem}.blue-kpi{padding:16px;min-height:108px}.process-panel{padding:16px}.process-metrics .blue-kpi{padding:12px;min-height:106px}.process-metrics .blue-kpi-value{font-size:23px}.process-head{font-size:14px}}
.bath-bubble {animation:bath-rise 3s linear infinite;}
@keyframes bath-rise {0%{transform:translate(0,8px);opacity:0}15%{opacity:.75}85%{opacity:.55}100%{transform:translate(3px,-75px);opacity:0}}
.bath-ripple {transform-box:fill-box;transform-origin:center;animation:bath-pop 2.4s ease-out infinite;}
@keyframes bath-pop {0%{transform:scale(.5);opacity:0}20%{opacity:.8}100%{transform:scale(1.7);opacity:0}}
@media(prefers-reduced-motion:reduce){.bath-bubble,.bath-ripple{animation:none;opacity:.55}}
</style>'''

def _html_text(value):
    # Keep the entire card in one Markdown HTML block. Raw blank lines terminate
    # that block even inside quoted attributes; encode them before rendering.
    return escape(str(value), quote=True).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "&#10;")


def kpi_html(label, value, help=None):
    tip = ''
    if help:
        accessible = _html_text(help)
        body = accessible.replace("&#10;", "<br>")
        tip = f'<span class="blue-kpi-help" tabindex="0" aria-label="{accessible}">?<span role="tooltip">{body}</span></span>'
    return f'<div class="blue-kpi"><div class="blue-kpi-label"><span>{_html_text(label)}</span>{tip}</div><div class="blue-kpi-value">{_html_text(value)}</div></div>'


def process_html(progress, current=None, risk="데이터 부족", alert="—", risk_help=None, alert_help=None, is_defect=False):
    p = max(0, min(100, int(progress)))
    def reading(key, decimals, unit=''):
        import math
        try:
            number = float(current[key])
            return f'{number:.{decimals}f}{unit}' if math.isfinite(number) else '—'
        except (TypeError, KeyError, ValueError):
            return '—'
    ph = reading('pH', 2)
    temp = reading('Temp', 1, ' °C')
    voltage = reading('Voltage', 1, ' V')
    sample = '' if current is None else _html_text(current.get('Timestamp', ''))
    cards = ''.join([
        kpi_html('진행률', f'{p}%'), kpi_html('pH', ph),
        kpi_html('AI 참고 위험도', risk, risk_help), kpi_html('온도', temp),
        kpi_html('경보 기준 초과 여부', alert, alert_help), kpi_html('전압', voltage),
    ])
    # Illustrative finish transition based on the supplied before/after photo.
    def finish(start, end):
        a = tuple(int(start[i:i+2], 16) for i in (1, 3, 5))
        b = tuple(int(end[i:i+2], 16) for i in (1, 3, 5))
        return '#' + ''.join(f'{round(x + (y-x)*p/100):02x}' for x, y in zip(a, b))
    metal_dark = finish('#454b50', '#929fa9')
    metal_light = finish('#93999d', '#ffffff')
    metal_mid = finish('#666d72', '#d8e2e9')
    metal_edge = finish('#3d444a', '#84939f')
    rim = finish('#8b969f', '#f1f6fa')
    wet = 0 < p < 100
    defect_surface = ''
    tarnish_opacity = max(0.0, min(1.0, (p - 50) / 50))
    if is_defect and tarnish_opacity > 0:
        defect_surface = f'<g opacity="{tarnish_opacity:.2f}" class="pipe-defect" mask="url(#pipe-surface-mask)" aria-hidden="true"><ellipse cx="3" cy="53" rx="12" ry="19" fill="url(#subtle-tarnish)"/><ellipse cx="17" cy="80" rx="18" ry="13" transform="rotate(35 17 80)" fill="url(#subtle-tarnish)"/><ellipse cx="53" cy="89" rx="19" ry="9" fill="url(#subtle-tarnish)" opacity=".55"/></g>'

    defect_note = ' · 불량 LOT 표면 표현(예시)' if is_defect else ''
    bubbles = '' 
    if wet:
        rising = ''.join(
            f'<circle class="bath-bubble" cx="{x}" cy="{y}" r="{r}" style="animation-delay:-{delay}s;animation-duration:{duration}s"/>'
            for x,y,r,delay,duration in [
                (276,252,3,0.2,3.4),(300,236,2,1.5,2.8),
                (323,255,3.5,0.8,3.2),(366,235,2,2.2,3.0),
                (389,258,2.8,1.1,3.7),(450,250,3.2,2.6,3.3),
                (474,230,2,0.5,2.7),(342,218,1.7,1.8,3.1),
            ]
        )
        surface = ''.join(
            f'<ellipse class="bath-ripple" cx="{x}" cy="175" rx="{r}" ry="2" style="animation-delay:-{delay}s"/>'
            for x,r,delay in [(281,6,0),(325,8,.7),(388,6,1.3),(449,9,1.8),(476,5,.4)]
        )
        bubbles = f'<g aria-hidden="true"><g clip-path="url(#bath-liquid-clip)" fill="#e6fff6" fill-opacity=".25" stroke="#e2fff3" stroke-width="1.2">{rising}</g><g fill="none" stroke="#edfff8" stroke-width="1.3">{surface}</g></g>'
    y = 152 if wet else 56
    state = '침지 중' if wet else ('인양 완료' if p == 100 else '침지 대기')
    return f'''<div class="process-panel"><div class="process-head"><span>파이프 표면처리 공정</span><span class="process-badge">{state} · {p}%</span></div>
<div class="process-layout"><svg viewBox="130 0 490 310" role="img" aria-label="{state}: 호이스트에 매달린 L자 파이프와 공정액 탱크의 개념도" xmlns="http://www.w3.org/2000/svg">
<defs><radialGradient id="subtle-tarnish"><stop offset="0" stop-color="#b38d58" stop-opacity=".28"/><stop offset=".5" stop-color="#bda27b" stop-opacity=".16"/><stop offset="1" stop-color="#cbb593" stop-opacity="0"/></radialGradient><mask id="pipe-surface-mask" maskUnits="userSpaceOnUse" x="-20" y="30" width="102" height="80"><path d="M0 30V63Q0 91 28 91H82" fill="none" stroke="white" stroke-width="28"/></mask><linearGradient id="metal" x1="0" x2="1"><stop stop-color="{metal_dark}"/><stop offset=".35" stop-color="{metal_light}"/><stop offset=".65" stop-color="{metal_mid}"/><stop offset="1" stop-color="{metal_edge}"/></linearGradient><linearGradient id="liquid" x2="0" y2="1"><stop stop-color="#b1dccd"/><stop offset="1" stop-color="#72ae9c"/></linearGradient><clipPath id="bath-liquid-clip"><rect x="234" y="176" width="270" height="92"/></clipPath></defs>
<ellipse cx="372" cy="286" rx="228" ry="10" fill="#eaf0f7"/>
<path d="M155 273V22H566V273" fill="none" stroke="#bbc9d9" stroke-width="9"/>
<rect x="311" y="13" width="75" height="24" rx="6" fill="#344e6e"/>
<path d="M349 37V{y}" stroke="#6d829a" stroke-width="3"/>
<path d="M231 143H507V271H231Z" fill="#f2f7fa" stroke="#98afc4" stroke-width="3"/>
<rect x="233" y="175" width="272" height="94" fill="url(#liquid)"/>
<g transform="translate(349,{y})"><path d="M0 0V30" stroke="#637d98" stroke-width="4"/><path d="M0 30V63Q0 91 28 91H82" fill="none" stroke="{metal_edge}" stroke-width="35"/><path d="M0 30V63Q0 91 28 91H82" fill="none" stroke="url(#metal)" stroke-width="29"/>{defect_surface}<ellipse cx="0" cy="30" rx="15" ry="6" fill="#324960" stroke="{rim}" stroke-width="3"/><ellipse cx="82" cy="91" rx="5" ry="15" fill="#324960" stroke="{rim}" stroke-width="3"/></g>
<rect x="233" y="175" width="272" height="94" fill="#73bba4" opacity=".34"/>
<path d="M233 175H505" stroke="#4d9781" stroke-width="2"/>{bubbles}
<path d="M231 143V271H507V143" fill="none" stroke="#7d97b0" stroke-width="4"/>
</svg><div class="process-metrics">{cards}</div></div><div class="process-note">측정 시각: {sample}{defect_note}</div></div>'''
