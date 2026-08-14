import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import random
import os
import base64
from datetime import datetime, time, timedelta
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from model_service import (lifestyle_profile, predict_efficiency, predict_lifestyle_risk,
                           predict_quality)

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="자니(Zzz-ni) 수면 건강 분석", page_icon="🌙", layout="wide")

def load_css(filename):
    """style.css를 읽어 <style>로 주입합니다. Streamlit에는 정적 CSS를 자동으로 붙여주는
    기능이 없어서, 파일로 분리하려면 이렇게 직접 읽어 넣어야 합니다."""
    root=Path(__file__).parent
    css=root.joinpath(filename).read_text(encoding="utf-8")
    background=root.joinpath("assets","wallpaperbetter.jpg")
    background_data=base64.b64encode(background.read_bytes()).decode() if background.exists() else ""
    css=css.replace("__SOMNIA_BACKGROUND__",background_data)
    st.markdown(f"<style>{css}</style>",unsafe_allow_html=True)

load_css("style.css")

music_path=Path(__file__).parent / "assets" / "pastoral-asher-fulero.mp3"
if music_path.exists():
    music_data=base64.b64encode(music_path.read_bytes()).decode()
    with st.container(key="bgm_player"):
        components.html(f"""
        <audio id="bgm" autoplay loop preload="auto">
          <source src="data:audio/mpeg;base64,{music_data}" type="audio/mpeg">
        </audio>
        <div class="volume"><span aria-hidden="true">♫</span><input id="volume" aria-label="배경음악 음량" type="range" min="0" max="1" step="0.01" value="0.35"></div>
        <style>
          html,body{{margin:0;background:transparent;overflow:hidden;font-family:sans-serif}}
          .volume{{height:28px;display:flex;align-items:center;gap:6px;padding:0 2px}}
          .volume span{{color:#fff;font-size:14px;line-height:1;text-shadow:0 1px 4px rgba(0,0,0,.5)}}
          input{{width:88px;height:12px;accent-color:#fff;cursor:pointer}}
        </style>
        <script>
          const audio=document.getElementById('bgm');
          const volume=document.getElementById('volume');
          audio.volume=0.35;
          volume.addEventListener('input',()=>{{audio.volume=Number(volume.value);audio.play().catch(()=>{{}});}});
          const start=()=>audio.play().catch(()=>{{}});
          start();
          document.addEventListener('pointerdown',start,{{once:true}});
          try{{window.parent.document.addEventListener('pointerdown',start,{{once:true}});}}catch(e){{}}
        </script>
        """,height=30,scrolling=False)

if "analyzed" not in st.session_state: st.session_state.analyzed=False
if "messages" not in st.session_state: st.session_state.messages=[{"role":"assistant","content":"안녕하세요! 분석 결과에 대해 궁금한 점을 물어보세요."}]
if st.session_state.get("chat_state_version")!="closed_by_default_v1":
    st.session_state.show_chat=False
    st.session_state.chat_state_version="closed_by_default_v1"
st.markdown('<div class="brand"><span class="mark">✦</span><span>자니<span class="blue">(Zzz-ni)</span></span></div>',unsafe_allow_html=True)

def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    if bmi < 25: return "Normal"
    if bmi < 30: return "Overweight"
    return "Obese"

def sleep_duration(bedtime,wake_time):
    base=datetime(2000,1,1)
    start=datetime.combine(base.date(),bedtime)
    end=datetime.combine(base.date(),wake_time)
    if end <= start: end += timedelta(days=1)
    return round((end-start).total_seconds()/3600,1)

PERSONALIZED_TIPS=[
    "주말에도 평일과 비슷한 시간에 일어나 수면 리듬을 일정하게 유지해 보세요.",
    "기상 후 15분 정도 햇빛을 쬐면 생체 시계를 안정시키는 데 도움이 됩니다.",
    "오후 2시 이후에는 커피와 에너지 음료를 줄여 깊은 수면을 준비해 보세요.",
    "잠들기 3시간 전부터는 음주를 피하고 물을 조금씩 마셔보세요.",
    "침실을 조용하고 어둡게 유지하고, 실내 온도는 약간 서늘하게 맞춰보세요.",
    "낮에 30분 정도 걷거나 가벼운 운동을 하되 취침 직전의 격한 운동은 피하세요.",
    "잠들기 전 5분 동안 천천히 호흡하거나 가볍게 스트레칭해 보세요.",
    "낮잠은 오후 3시 이전에 20분 이내로 짧게 자는 것이 좋습니다.",
    "침대에서는 업무나 영상 시청을 줄이고 수면을 위한 공간으로 사용해 보세요.",
    "잠들기 전 걱정되는 일을 메모해 두고 생각을 잠시 내려놓는 습관을 만들어 보세요.",
]

def save_and_analyze(data):
    st.session_state.data=data
    st.session_state.personalized_tip=random.choice(PERSONALIZED_TIPS)
    st.session_state.inline_coaching=None
    st.session_state.analyzed=True
    st.rerun()

def ask_sleep_coach(prompt):
    st.session_state.messages.append({"role":"user","content":prompt})
    d=st.session_state.get("data",{})
    api_key=os.getenv("OPENAI_API_KEY","").strip()
    if not api_key:
        answer="OpenAI API 키가 아직 설정되지 않았어요. 프로젝트의 .env 파일에 OPENAI_API_KEY를 입력한 뒤 앱을 다시 실행해 주세요."
    else:
        try:
            health_context=(
                f"사용자 수면 데이터: 수면 {d.get('sleep','미입력')}시간, "
                f"스트레스 {d.get('stress','미입력')}/10, 신체 활동 {d.get('activity','미입력')}분/일, "
                f"BMI {d.get('bmi','미입력')}, 카페인 {d.get('caffeine','미입력')}잔, "
                f"최근 24시간 내 음주 {d.get('recent_alcohol','미입력')}, 휴대폰 {d.get('phone_hours','미입력')}시간, "
                f"낮 시간 졸림 {d.get('daytime_sleepiness','미입력')}/10."
            )
            client=OpenAI(api_key=api_key)
            response=client.responses.create(
                model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),
                instructions=("당신은 친절한 한국어 AI 수면 코치입니다. 의료 진단을 내리지 말고, 생활 습관 개선에 도움이 되는 일반적인 정보만 간결하게 안내하세요. 응급 증상이나 심각한 수면장애가 의심되면 의료진 상담을 권하세요. "+health_context),
                input=[{"role":m["role"],"content":m["content"]} for m in st.session_state.messages[-10:]],
            )
            answer=response.output_text or "답변을 생성하지 못했어요. 잠시 후 다시 시도해 주세요."
        except Exception:
            answer="현재 AI 코치에 연결할 수 없어요. API 키와 인터넷 연결을 확인한 뒤 다시 시도해 주세요."
    st.session_state.messages.append({"role":"assistant","content":answer})
    return answer

def lifestyle_radar(d):
    """생활습관 프로필 레이더. 축마다 방향을 맞춰 바깥쪽일수록 양호합니다."""
    profile=lifestyle_profile(d)
    if not profile: return None
    axes=[p["axis"] for p in profile]+[profile[0]["axis"]]
    scores=[p["score"] for p in profile]+[profile[0]["score"]]
    fig=go.Figure(go.Scatterpolar(r=scores,theta=axes,fill="toself",mode="lines+markers",
        line=dict(color="#2265e5",width=2),fillcolor="rgba(34,101,229,.18)",marker=dict(size=7)))
    fig.update_layout(title=dict(text="생활습관 프로필 · 바깥쪽일수록 양호",x=.045,xanchor="left",font=dict(size=20)),
        height=380,margin=dict(l=60,r=60,t=76,b=40),showlegend=False,
        polar=dict(bgcolor="white",radialaxis=dict(range=[0,100],tickvals=[25,50,75,100],ticksuffix="",tickfont=dict(size=11,color="#5a6d7a"),gridcolor="#e6edf1"),
                   angularaxis=dict(tickfont=dict(size=14,color="#14283a"),gridcolor="#e6edf1")),
        paper_bgcolor="white",font=dict(family="Pretendard",color="#14283a"))
    return fig

def status_metric(container,label,value,status,tone="good"):
    container.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><span class="pill pill-{tone}">{status}</span></div>',unsafe_allow_html=True)

if not st.session_state.analyzed:
    st.markdown('<div class="eyebrow">AI SLEEP HEALTH ANALYSIS</div><div class="hero">당신의 수면을<br><span class="blue">측정해 드립니다.</span></div><div class="sub">필수 항목을 모두 입력하면 수면 건강 분석을 시작할 수 있어요.</div>',unsafe_allow_html=True)
    st.markdown('<div class="form-title">수면 건강 측정</div>',unsafe_allow_html=True)

    def field_label(text,required=True):
        star='<span style="color:#e5484d;font-weight:900">*</span>' if required else '<span style="color:#758797;font-size:.82rem;font-weight:600">(선택)</span>'
        st.markdown(f'<div style="font-size:1rem;font-weight:700;margin:.2rem 0 .35rem">{text} {star}</div>',unsafe_allow_html=True)

    with st.form("sleep_health_form"):
        basic_col,extra_col=st.columns(2,gap="large")
        with basic_col:
            st.markdown('<div class="detail-section-title">기본 정보</div>',unsafe_allow_html=True)
            field_label("성별")
            d_gender=st.selectbox("성별",["Female","Male"],index=None,placeholder="성별을 선택하세요",format_func=lambda x:{"Female":"여성","Male":"남성"}[x],label_visibility="collapsed")
            field_label("나이")
            d_age=st.number_input("나이",min_value=18,max_value=100,value=None,step=1,placeholder="나이를 입력하세요",label_visibility="collapsed")
            height_col,weight_col=st.columns(2)
            with height_col:
                field_label("키 (cm)")
                d_height=st.number_input("키 (cm)",min_value=120.0,max_value=220.0,value=None,step=.5,placeholder="키",key="detail_height",label_visibility="collapsed")
            with weight_col:
                field_label("몸무게 (kg)")
                d_weight=st.number_input("몸무게 (kg)",min_value=30.0,max_value=200.0,value=None,step=.5,placeholder="몸무게",key="detail_weight",label_visibility="collapsed")
            d_bmi=round(d_weight/((d_height/100)**2),1) if d_height is not None and d_weight is not None else None
            st.markdown('<div style="height:20px"></div>',unsafe_allow_html=True)
            with st.expander("선택 입력 항목  ·  혈압, 심박수, 하루 걸음 수"):
                st.caption("알고 있는 항목만 입력해 주세요. 입력하지 않아도 분석할 수 있습니다.")
                bp_sys_col,bp_dia_col=st.columns(2)
                with bp_sys_col:
                    field_label("수축기 혈압 (mmHg)",False)
                    d_sys=st.number_input("수축기 혈압 (mmHg)",min_value=70,max_value=250,value=None,step=1,placeholder="예: 120",help="혈압계에 크게 표시되는 위쪽 숫자입니다.",label_visibility="collapsed")
                with bp_dia_col:
                    field_label("이완기 혈압 (mmHg)",False)
                    d_dia=st.number_input("이완기 혈압 (mmHg)",min_value=40,max_value=150,value=None,step=1,placeholder="예: 80",help="혈압계에 표시되는 아래쪽 숫자입니다.",label_visibility="collapsed")
                field_label("심박수 (회/분)",False)
                d_hr=st.number_input("심박수 (회/분)",min_value=40,max_value=200,value=None,step=1,placeholder="안정 시 심박수",help="안정 시 심박수를 입력하세요.",label_visibility="collapsed")
                field_label("하루 걸음 수",False)
                d_steps=st.number_input("하루 걸음 수",min_value=0,max_value=50000,value=None,step=500,placeholder="하루 평균 걸음 수",label_visibility="collapsed")
        with extra_col:
            st.markdown('<div class="detail-section-title">상세 건강 정보</div>',unsafe_allow_html=True)
            field_label("신체 활동수준 (분/일)")
            d_activity=st.number_input("신체 활동수준 (분/일)",min_value=0,max_value=180,value=None,step=5,placeholder="하루 활동 시간을 입력하세요",label_visibility="collapsed")
            field_label("취침시간 / 기상시간")
            sleep_scale_start=datetime(2000,1,1,12,0)
            sleep_scale_end=datetime(2000,1,2,12,0)
            sleep_range=st.slider(
                "취침시간 / 기상시간",
                min_value=sleep_scale_start,
                max_value=sleep_scale_end,
                value=(datetime(2000,1,1,23,30),datetime(2000,1,2,6,30)),
                step=timedelta(minutes=15),
                format="HH:mm",
                key="detail_sleep_range_v1",
                label_visibility="collapsed",
            )
            d_bedtime=sleep_range[0].time()
            d_wake_time=sleep_range[1].time()
            st.caption(f"취침 {d_bedtime.strftime('%H:%M')}  ·  기상 {d_wake_time.strftime('%H:%M')}")
            field_label("스트레스 지수 (1~10)")
            d_stress=st.slider("스트레스 지수 (1~10)",min_value=1,max_value=10,value=5,step=1,label_visibility="collapsed")
            field_label("하루 카페인 섭취량 (잔)")
            d_caffeine=st.number_input("하루 카페인 섭취량 (잔)",min_value=0.0,max_value=15.0,value=None,step=.5,placeholder="섭취하지 않으면 0",help="커피·에너지 음료·카페인 차를 합산해 입력하세요.",label_visibility="collapsed")
            field_label("하루 휴대폰 사용시간 (시간)")
            d_phone=st.number_input("하루 휴대폰 사용시간 (시간)",min_value=0.0,max_value=24.0,value=None,step=.5,placeholder="사용하지 않으면 0",help="하루 평균 휴대폰 사용시간을 입력하세요.",label_visibility="collapsed")
            smoking_col,alcohol_col=st.columns(2)
            with smoking_col:
                field_label("흡연 여부")
                d_smoking=st.radio("흡연 여부",["비흡연","흡연"],index=None,horizontal=True,label_visibility="collapsed")
            with alcohol_col:
                field_label("최근 24시간 내 음주 여부")
                d_recent_alcohol=st.radio("최근 24시간 내 음주 여부",["음주 안 함","음주함"],index=None,horizontal=True,label_visibility="collapsed")
            field_label("낮 시간 졸림 정도 (1~10단계)")
            d_daytime_sleepiness=st.slider("낮 시간 졸림 정도 (1~10단계)",min_value=1,max_value=10,value=5,step=1,help="1 = 전혀 졸리지 않음, 5 = 가끔 졸림, 10 = 매우 심하게 졸림",label_visibility="collapsed")
        if st.form_submit_button("분석 시작하기 →",use_container_width=True):
            required_values=[
                ("성별",d_gender),("나이",d_age),("키",d_height),("몸무게",d_weight),
                ("취침시간",d_bedtime),("기상시간",d_wake_time),("신체 활동수준",d_activity),
                ("스트레스 지수",d_stress),("카페인 섭취량",d_caffeine),("휴대폰 사용시간",d_phone),
                ("흡연 여부",d_smoking),("음주 여부",d_recent_alcohol),("낮 시간 졸림 정도",d_daytime_sleepiness),
            ]
            missing=[label for label,value in required_values if value is None]
            if missing:
                st.error("필수 입력값을 모두 입력해 주세요: " + ", ".join(missing))
                st.stop()
            if (d_sys is None)!=(d_dia is None):
                st.error("혈압을 입력하려면 수축기와 이완기 혈압을 모두 입력해 주세요.")
                st.stop()
            # 뒤바꿔 입력한 경우는 각 항목의 범위만으로 걸러지지 않음
            if d_sys is not None and d_dia is not None and d_sys<=d_dia:
                st.error(f"혈압을 다시 확인해 주세요. 수축기({d_sys})는 이완기({d_dia})보다 높아야 합니다.")
                st.stop()
            d_sleep=sleep_duration(d_bedtime,d_wake_time)
            save_and_analyze({"mode":"상세 폼","gender":d_gender,"age":d_age,"occupation":None,"height":d_height,"weight":d_weight,"bedtime":d_bedtime.strftime("%H:%M"),"wake_time":d_wake_time.strftime("%H:%M"),"sleep":d_sleep,"quality":None,"activity":d_activity,"stress":d_stress,"bmi":d_bmi,"sys":d_sys,"dia":d_dia,"heart_rate":d_hr,"daily_steps":d_steps,"sleep_disorder":"None","caffeine":d_caffeine,"recent_alcohol":d_recent_alcohol,"phone_hours":d_phone,"daytime_sleepiness":d_daytime_sleepiness,"smoking":d_smoking})
else:
    d=st.session_state.data
    # 화면에 나가는 진단값은 전부 모델에서만 옵니다. 모델이 없거나 입력이 모자라면
    # 임의 수식으로 채우지 않고 안내를 띄웁니다.
    efficiency=predict_efficiency(d)
    verdict=predict_quality(d)
    lifestyle=predict_lifestyle_risk(d)
    st.markdown(f'<div class="eyebrow">YOUR SLEEP REPORT · {d["mode"]}</div><div class="hero">수면 건강 분석이<br><span class="blue">완료되었어요.</span></div><div class="sub">입력한 생활 습관을 기반으로 현재 수면 상태와 주요 위험 요인을 분석했습니다.</div>',unsafe_allow_html=True)

    # 나의 수면 분석 — 모델 3개의 결과를 한 줄로 요약합니다.
    def summary_tile(label,value,tone,caption):
        return f'<div class="tile tile-{tone}"><div class="tile-label">{label}</div><div class="tile-value">{value}</div><div class="tile-caption">{caption}</div></div>'
    tiles=[]
    if verdict and verdict["score"] is not None:
        tiles.append(summary_tile("수면의 질",f'{verdict["score"]} <span class="tile-unit">/ 10</span>',verdict["tone"],verdict["text"]))
    else:
        tiles.append(summary_tile("수면의 질","—","idle","심박수·혈압·걸음 수 필요"))
    if efficiency:
        caption={"good":"정상 (85% 이상)","warn":"주의 (75~85%)","bad":"낮음 (75% 미만)"}[efficiency["tone"]]
        tiles.append(summary_tile("수면 효율",f'{efficiency["value"]:.1f}<span class="tile-unit">%</span>',efficiency["tone"],caption))
    else:
        tiles.append(summary_tile("수면 효율","—","idle","입력이 모자랍니다"))
    if lifestyle:
        tiles.append(summary_tile("생활습관 위험도",lifestyle["level"],lifestyle["tone"],f'지수 {lifestyle["value"]:.2f}'))
    else:
        tiles.append(summary_tile("생활습관 위험도","—","idle","입력이 모자랍니다"))
    st.markdown(f'<div class="summary-title">나의 수면 분석</div><div class="tiles">{"".join(tiles)}</div>',unsafe_allow_html=True)

    left,right=st.columns([1.55,1])
    with left:
        radar=lifestyle_radar(d)
        if radar:
            st.plotly_chart(radar,use_container_width=True,config={"displayModeBar":False})
        else:
            st.markdown('<div class="pending"><div class="pending-title">생활습관 프로필</div><div class="pending-body">생활습관 입력이 모자라 프로필을 그릴 수 없습니다.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="notice">이 결과는 의료 진단이 아닌 건강 관리 참고용입니다. 증상이 지속되면 전문의와 상담하세요.</div>',unsafe_allow_html=True)
    with right:
        # 효율은 학습 범위가 57~99라 0부터 그리면 바늘이 항상 오른쪽에 몰립니다.
        if efficiency:
            g=efficiency["gauge"]
            axis_min,axis_max,warn,good=g["min"],g["max"],g["warn"],g["good"]
            value=efficiency["value"]
            score_color="#d94b4b" if value<warn else ("#e58a1f" if value<good else "#15966f")
            gauge=go.Figure(go.Indicator(mode="gauge+number",value=value,title={"text":"수면 효율","font":{"size":20,"color":"#14283a"}},number={"suffix":"%","font":{"color":score_color,"size":46}},gauge={"axis":{"range":[axis_min,axis_max],"tickfont":{"size":13,"color":"#5a6d7a"}},"bar":{"color":score_color,"thickness":.68},"bgcolor":"#eef2f3","borderwidth":0,"steps":[{"range":[axis_min,warn],"color":"#f9dada"},{"range":[warn,good],"color":"#fde9c9"},{"range":[good,axis_max],"color":"#d9f1e7"}]}))
            gauge.update_layout(height=340,margin=dict(l=48,r=48,t=78,b=28),paper_bgcolor="white",font=dict(family="Pretendard",color="#14283a"))
            st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
        else:
            st.markdown('<div class="pending"><div class="pending-title">수면 효율</div><div class="pending-body">효율 예측 모델을 실행할 수 없습니다.<br>입력이 모두 채워졌는지 확인해 주세요.</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="verdict-label">현재 상태</div>',unsafe_allow_html=True)
        if verdict:
            st.markdown(f'<div class="verdict verdict-{verdict["tone"]}"><span>{verdict["icon"]}</span><span>{verdict["text"]}</span></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="verdict-note">수면의 질 예측 모델 · 확신도 {max(verdict["proba"].values()):.0%}</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="pending pending-verdict"><div class="pending-body">수면의 질 모델에는 선택 항목인<br>심박수·혈압·걸음 수가 필요합니다.</div></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:28px"></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    status_metric(a,"수면 시간",f"{d['sleep']}시간","권장보다 짧음" if d['sleep']<7 else "적정 범위","warn" if d['sleep']<7 else "good")
    status_metric(b,"스트레스",f"{d['stress']}/10","관리 필요" if d['stress']>6 else "안정적","warn" if d['stress']>6 else "good")
    if d["mode"]=="상세 폼":
        status_metric(c,"흡연 여부",d["smoking"],d["smoking"],"bad" if d["smoking"]=="흡연" else "good")
    else:
        bmi_status=bmi_category(d['bmi'])
        status_metric(c,"BMI",f"{d['bmi']:.1f}",bmi_status,"good" if bmi_status=="Normal" else ("bad" if bmi_status=="Obese" else "warn"))
    if d["mode"]=="상세 폼":
        st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
        c4,c5=st.columns(2)
        status_metric(c4,"하루 카페인",f"{d['caffeine']:g}잔","섭취 조절 권장" if d['caffeine']>2 else "적정 수준","warn" if d['caffeine']>2 else "good")
        status_metric(c5,"최근 24시간 내 음주",d['recent_alcohol'],"수면 영향 가능" if d['recent_alcohol']=="음주함" else "음주 없음","warn" if d['recent_alcohol']=="음주함" else "good")
    if d["mode"]=="상세 폼":
        with st.expander("입력한 상세 데이터 확인"):
            # 혈압·심박수·걸음 수는 선택 항목이라 비어 있을 수 있습니다.
            blood_pressure_text=f'{d["sys"]}/{d["dia"]} mmHg' if d["sys"] is not None and d["dia"] is not None else "미입력"
            heart_rate_text=f'{d["heart_rate"]}회/분' if d["heart_rate"] is not None else "미입력"
            daily_steps_text=f'{d["daily_steps"]:,}걸음' if d["daily_steps"] is not None else "미입력"
            details=[
                ("성별","여성" if d["gender"]=="Female" else "남성"),("나이",f'{d["age"]}세'),
                ("키",f'{d.get("height",0):g}cm'),("몸무게",f'{d.get("weight",0):g}kg'),
                ("BMI",f'{d["bmi"]:.1f}'),
                ("수면 패턴",f'{d["bedtime"]} ~ {d["wake_time"]}'),("자동 계산 수면시간",f'{d["sleep"]}시간'),
                ("신체 활동수준",f'{d["activity"]}분/일'),("스트레스 지수",f'{d["stress"]}/10'),
                ("혈압",blood_pressure_text),("심박수",heart_rate_text),
                ("하루 걸음 수",daily_steps_text),("하루 카페인 섭취량",f'{d["caffeine"]:g}잔'),
                ("최근 24시간 내 음주 여부",d["recent_alcohol"]),("하루 휴대폰 사용시간",f'{d.get("phone_hours",0):g}시간'),
                ("낮 시간 졸림 정도",f'{d.get("daytime_sleepiness",0)}/10'),("흡연 여부",d["smoking"]),
            ]
            rows="".join(f'<div class="dl-row"><span class="dl-k">{label}</span><span class="dl-v">{value}</span></div>' for label,value in details)
            st.markdown(f'<div class="dl">{rows}</div>',unsafe_allow_html=True)
    personalized_tip=st.session_state.get("personalized_tip") or random.choice(PERSONALIZED_TIPS)
    st.session_state.personalized_tip=personalized_tip
    st.markdown(f'<div class="tip"><div class="eyebrow">PERSONALIZED TIP</div><h3>오늘부터 이렇게 시작해 보세요</h3><p>{personalized_tip}</p></div>',unsafe_allow_html=True)
    st.markdown('<div style="height:32px"></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    if c1.button("← 정보 다시 입력하기",use_container_width=True): st.session_state.analyzed=False; st.rerun()
    if c2.button("✦ 내 결과를 바탕으로 AI 수면 코칭 받기",use_container_width=True):
        with st.spinner("AI 수면 코치가 결과를 확인하고 있어요..."):
            st.session_state.inline_coaching=ask_sleep_coach("내 수면 분석 결과를 바탕으로 가장 개선이 필요한 부분과 오늘부터 실천할 수 있는 방법 3가지를 알려줘.")
    if st.session_state.get("inline_coaching"):
        with st.container(key="inline_coaching"):
            st.markdown("#### ✦ 내 결과 기반 AI 수면 코칭")
            st.markdown(st.session_state.inline_coaching)

st.markdown('<div class="foot">TEAM ||&nbsp;&nbsp;잠은 죽어서 자조</div>',unsafe_allow_html=True)

with st.container(key="chat_fab"):
    with st.popover("💬",help="AI 수면 코치 열기"):
        st.markdown("#### ✦ AI 수면 코치")
        with st.container(height=320,border=False):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]): st.write(message["content"])
        if prompt:=st.chat_input("궁금한 점을 입력하세요"):
            ask_sleep_coach(prompt)
            st.rerun()
