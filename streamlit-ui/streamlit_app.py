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

def setting(name,default=""):
    """설정값을 st.secrets -> 환경변수(.env) 순으로 찾습니다.

    로컬은 .env, .env를 올릴 수 없는 배포 환경은 secrets를 씁니다.
    secrets 파일이 없으면 st.secrets 접근 자체가 예외라 감싸 둡니다.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name,default).strip()

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
          input{{width:100px;height:12px;accent-color:#fff;cursor:pointer}}
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
    api_key=setting("OPENAI_API_KEY")
    if not api_key:
        answer="OpenAI API 키가 아직 설정되지 않았어요. 로컬은 .env, 배포 환경은 Secrets에 OPENAI_API_KEY를 넣은 뒤 앱을 다시 실행해 주세요."
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
                model=setting("OPENAI_MODEL","gpt-4.1-mini"),
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
    st.markdown("""
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {
        --primary-color: #2d70ee !important;
    }

    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { gap: .45rem !important; }
    [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: .5rem !important; }
    .form-title {
        color: #10213d !important;
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        line-height: 1.25 !important;
        margin: 0 0 .25rem !important;
        text-shadow: none !important;
    }
    .form-guide {
        color: #75839a !important;
        font-size: .82rem !important;
        margin: 0 0 .75rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.90) !important;
        border: 1px solid rgba(190,205,225,.55) !important;
        border-radius: 18px !important;
        padding: 10px 12px !important;
        box-shadow: 0 5px 18px rgba(20,50,100,.06) !important;
        backdrop-filter: none !important;
    }
    .st-key-basic_info_card,
    .st-key-sleep_habit_card,
    .st-key-activity_card,
    .st-key-stress_card,
    .st-key-health_detail_card,
    .st-key-sleepiness_card {
        min-height: 0 !important;
        margin-bottom: 0 !important;
        gap: .42rem !important;
        padding: 10px 12px !important;
        background: rgba(255,255,255,.90) !important;
        border: 1px solid rgba(190,205,225,.55) !important;
        border-radius: 14px !important;
        box-shadow: 0 5px 18px rgba(20,50,100,.06) !important;
    }
    .st-key-basic_info_card {
        padding-bottom: 28px !important;
    }
    /* 좌우 입력 열의 시작점과 끝점을 동일하게 맞춥니다. */
    [data-testid="stColumn"]:has(.st-key-basic_info_card),
    [data-testid="stColumn"]:has(.st-key-health_detail_card) {
        display: flex !important;
    }
    [data-testid="stColumn"]:has(.st-key-basic_info_card) > [data-testid="stVerticalBlock"],
    [data-testid="stColumn"]:has(.st-key-health_detail_card) > [data-testid="stVerticalBlock"] {
        flex: 1 1 auto !important;
        height: 100% !important;
    }
    [data-testid="stColumn"]:has(.st-key-basic_info_card) > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-basic_info_card),
    [data-testid="stColumn"]:has(.st-key-health_detail_card) > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(.st-key-health_detail_card) {
        flex: 1 1 auto !important;
    }
    .st-key-basic_info_card,
    .st-key-health_detail_card {
        height: 100% !important;
    }

    .st-key-basic_info_card [data-testid="stVerticalBlock"],
    .st-key-sleep_habit_card [data-testid="stVerticalBlock"],
    .st-key-activity_card [data-testid="stVerticalBlock"],
    .st-key-stress_card [data-testid="stVerticalBlock"],
    .st-key-health_detail_card [data-testid="stVerticalBlock"],
    .st-key-sleepiness_card [data-testid="stVerticalBlock"] {
        gap: .42rem !important;
    }
    .section-head {
        display:flex !important;
        align-items:center !important;
        gap:10px !important;
        min-height:36px !important;
        margin-bottom:7px !important;
    }
    .st-key-sleep_habit_card .section-head,
    .st-key-health_detail_card .section-head {
        margin-bottom:15px !important;
    }
    .section-icon {
        width:32px !important; height:32px !important; min-width:32px !important;
        display:inline-flex !important; align-items:center !important; justify-content:center !important;
        border-radius:50% !important;
    }
    .section-icon svg { width:18px !important; height:18px !important; fill:none; stroke:currentColor; stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round; }
    .section-icon.blue { background:#e9f1ff !important; color:#2f6ff2 !important; }
    .section-icon.purple { background:#f0ebff !important; color:#7257f5 !important; }
    .section-icon.green { background:#e3f7ef !important; color:#22a87a !important; }
    .section-icon.orange { background:#fff0e1 !important; color:#f0802d !important; }
    .section-icon.sky { background:#e7f2ff !important; color:#3479ed !important; }
    .section-title {
        color: #13294b !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        margin: 0 0 1px !important;
        padding: 0 !important;
        text-shadow: none !important;
    }
    .section-desc {
        color: #75839a !important;
        font-size: 11px !important;
        line-height: 1.35 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stTimeInput"] input {
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInputContainer"] input,
    div[data-testid="stTimeInput"] input,
    div[data-baseweb="select"] > div {
        min-height: 34px !important;
        height: 34px !important;
        background: rgba(255,255,255,.96) !important;
    }
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stNumberInputContainer"] button,
    div[data-testid="stTimeInput"] > div,
    div[data-baseweb="select"] > div { min-height:34px !important; height:34px !important; }
    label[data-testid="stWidgetLabel"] p {
        color: #13294b !important;
        font-size: 12px !important;
        font-weight: 650 !important;
    }
    .st-key-basic_info_card label[data-testid="stWidgetLabel"],
    .st-key-health_detail_card label[data-testid="stWidgetLabel"] {
        padding-left: 5px !important;
    }
    .st-key-sleep_habit_card label[data-testid="stWidgetLabel"] {
        padding-left: 5px !important;
    }

    .bmi-card {
        background: linear-gradient(90deg,#eef5ff,#f7fbff) !important;
        border: 1px solid #cfe0ff !important;
        border-radius: 12px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        padding: 7px 10px !important;
        margin-top: 4px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 8px !important;
        min-height: 44px !important;
        overflow: hidden !important;
    }
    .bmi-title { color:#183963 !important; font-size:13px !important; font-weight:700 !important; }
    .bmi-sub { color:#8390a3 !important; font-size:10px !important; }
    .bmi-value-wrap {
        display:flex !important;
        align-items:center !important;
        justify-content:flex-end !important;
        gap:7px !important;
        min-width:0 !important;
        flex-shrink:1 !important;
    }
    .bmi-value {
        color:#2864dc !important;
        font-size:20px !important;
        font-weight:750 !important;
        line-height:1 !important;
        white-space:nowrap !important;
    }
    .bmi-badge {
        background:#e1f6e9 !important;
        color:#32a064 !important;
        border-radius:999px !important;
        padding:3px 6px !important;
        font-size:9px !important;
        font-weight:700 !important;
        line-height:1.2 !important;
        white-space:nowrap !important;
    }
    .scale-right { color:#75839a !important; font-size:11px !important; text-align:right !important; }
    [data-testid="stCaptionContainer"] p { font-size:11px !important; }

    div[data-testid="stFormSubmitButton"] {
        margin-top: 14px !important;
    }
    div[data-testid="stFormSubmitButton"] button,
    div.stButton > button {
        height: 36px !important;
        min-height: 36px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        border: 0 !important;
        border-radius: 999px !important;
        background: linear-gradient(90deg,#2563eb,#3087e8) !important;
        color: white !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 18px rgba(37,99,235,.22) !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover,
    div.stButton > button:hover {
        filter: brightness(1.04) !important;
        color: white !important;
    }
    .submit-gap { height: 2px !important; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="form-title">핵심 생활 습관 입력</div><div class="form-guide">정확한 분석을 위해 정보를 입력해주세요.</div>',unsafe_allow_html=True)

    def section_header(icon,title,description,tone="blue"):
        icon_svg={
            "기본 정보":'<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4.5 21a7.5 7.5 0 0 1 15 0"/></svg>',
            "상세 건강 정보":'<svg viewBox="0 0 24 24"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/><path d="M7 12h3l1-2 2 4 1-2h3"/></svg>',
            "수면 습관":'<svg viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>',
            "신체 활동 수준":'<svg viewBox="0 0 24 24"><circle cx="14" cy="4" r="2"/><path d="m5 21 3-5 2-4 4 3 3 6M7 10l3-2 3 2 3-2M11 12l-2 4"/></svg>',
            "스트레스 지수":'<svg viewBox="0 0 24 24"><path d="M9.5 4A3.5 3.5 0 0 0 6 7.5a3 3 0 0 0-1 5.8A3.5 3.5 0 0 0 8.5 19H10V5.5A1.5 1.5 0 0 0 8.5 4M14.5 4A3.5 3.5 0 0 1 18 7.5a3 3 0 0 1 1 5.8 3.5 3.5 0 0 1-3.5 5.7H14V5.5A1.5 1.5 0 0 1 15.5 4M6 9h4M14 9h4M6 15h4M14 15h4"/></svg>',
            "낮 시간 졸림 정도 (1~10단계)":'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.41M17.66 6.34l1.41-1.41"/></svg>',
        }.get(title,'<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/></svg>')
        st.markdown(
            f'<div class="section-head"><span class="section-icon {tone}">{icon_svg}</span>'
            f'<div><div class="section-title">{title}</div><div class="section-desc">{description}</div></div></div>',
            unsafe_allow_html=True,
        )

    with st.form("sleep_health_form",border=False):
        left_col,right_col=st.columns(2,gap="medium")
        with left_col:
            with st.container(border=True,key="basic_info_card"):
                section_header("👤","기본 정보","기본적인 신체 정보를 입력해 주세요.","blue")
                gender_col,age_col=st.columns(2)
                with gender_col:
                    d_gender=st.selectbox("성별",["Female","Male"],index=0,format_func=lambda x:{"Female":"여성","Male":"남성"}[x])
                with age_col:
                    d_age=st.number_input("나이",min_value=1,max_value=120,value=32,step=1)
                height_col,weight_col=st.columns(2)
                with height_col:
                    d_height=st.number_input("키 (cm)",min_value=100.0,max_value=250.0,value=170.0,step=.1,format="%.2f",key="detail_height")
                with weight_col:
                    d_weight=st.number_input("몸무게 (kg)",min_value=20.0,max_value=300.0,value=65.0,step=.1,format="%.2f",key="detail_weight")
                d_bmi=round(d_weight/((d_height/100)**2),1) if d_height is not None and d_weight is not None else None
                if d_bmi is None:
                    bmi_value,bmi_text="—","입력 대기"
                elif d_bmi < 18.5:
                    bmi_value,bmi_text=f"{d_bmi:.1f}","저체중"
                elif d_bmi < 23:
                    bmi_value,bmi_text=f"{d_bmi:.1f}","정상 범위"
                elif d_bmi < 25:
                    bmi_value,bmi_text=f"{d_bmi:.1f}","과체중"
                else:
                    bmi_value,bmi_text=f"{d_bmi:.1f}","비만 범위"
                st.markdown(f'<div class="bmi-card"><div><div class="bmi-title">BMI</div><div class="bmi-sub">체질량지수</div></div><div class="bmi-value-wrap"><div class="bmi-value">{bmi_value}</div><div class="bmi-badge">{bmi_text}</div></div></div>',unsafe_allow_html=True)
            with st.container(border=True,key="sleep_habit_card"):
                section_header("🌙","수면 습관","하루 수면 패턴을 입력해 주세요.","purple")
                # 효율 모델의 필수 입력. 비면 효율 예측이 통째로 비활성화됩니다.
                bedtime_col,wake_col,awake_col=st.columns(3)
                with bedtime_col:
                    d_bedtime=st.time_input("취침시간",value=time(23,30),step=timedelta(minutes=15),key="detail_bedtime_input")
                with wake_col:
                    d_wake_time=st.time_input("기상시간",value=time(6,30),step=timedelta(minutes=15),key="detail_wake_time_input")
                with awake_col:
                    d_awakenings=st.number_input("밤중 깬 횟수",min_value=0,max_value=10,value=2,step=1,key="detail_night_awakenings",help="자는 동안 잠에서 깬 횟수입니다. 학습 데이터 기준 0~10회, 중앙값 2회.")
            with st.container(border=True,key="activity_card"):
                activity_title_col,activity_input_col=st.columns([1.25,1])
                with activity_title_col:
                    section_header("🏃","신체 활동 수준","일반적인 하루 활동량을 입력해 주세요.","green")
                with activity_input_col:
                    d_activity=st.number_input("활동 시간 (분/일)",min_value=0,max_value=1440,value=45,step=5,label_visibility="collapsed")
            with st.container(border=True,key="stress_card"):
                section_header("🧠","스트레스 지수","평소 스트레스 정도를 1~10단계로 평가해 주세요.","orange")
                d_stress=st.slider("스트레스 지수 (1~10)",min_value=1,max_value=10,value=5,step=1,label_visibility="collapsed")
                scale_left,scale_right=st.columns(2)
                scale_left.caption("1 : 전혀 스트레스 없음")
                scale_right.markdown('<div class="scale-right">10 : 매우 높은 스트레스</div>',unsafe_allow_html=True)
        with right_col:
            with st.container(border=True,key="health_detail_card"):
                section_header("💙","상세 건강 정보","더 정확한 분석을 위한 건강 정보를 입력해 주세요.","blue")
                bp_sys_col,bp_dia_col=st.columns(2)
                with bp_sys_col:
                    d_sys=st.number_input("수축기 혈압 (mmHg)",min_value=50,max_value=250,value=120,step=1)
                with bp_dia_col:
                    d_dia=st.number_input("이완기 혈압 (mmHg)",min_value=30,max_value=200,value=80,step=1)
                d_hr=st.number_input("심박수 (회/분)",min_value=30,max_value=220,value=70,step=1)
                d_steps=st.number_input("하루 걸음 수",min_value=0,max_value=100000,value=7000,step=100)
                caffeine_col,phone_col=st.columns(2)
                with caffeine_col:
                    d_caffeine=st.number_input("하루 카페인 섭취량 (잔)",min_value=0.0,max_value=20.0,value=1.0,step=.5,format="%.2f")
                with phone_col:
                    d_phone=st.number_input("하루 휴대폰 사용시간 (시간)",min_value=0.0,max_value=24.0,value=5.0,step=.5,format="%.2f")
                smoking_col,alcohol_col=st.columns(2)
                with smoking_col:
                    d_smoking=st.radio("흡연 여부",["비흡연","흡연"],horizontal=True)
                with alcohol_col:
                    d_recent_alcohol=st.radio("최근 24시간 내 음주 여부",["음주 안 함","음주함"],horizontal=True)
            with st.container(border=True,key="sleepiness_card"):
                section_header("☀️","낮 시간 졸림 정도 (1~10단계)","평소 낮 시간의 졸림 정도를 평가해 주세요.","sky")
                d_daytime_sleepiness=st.slider("낮 시간 졸림 정도 (1~10단계)",min_value=1,max_value=10,value=5,step=1,help="1 = 전혀 졸리지 않음, 5 = 가끔 졸림, 10 = 매우 심하게 졸림",label_visibility="collapsed")
                scale_left,scale_right=st.columns(2)
                scale_left.caption("1 : 전혀 졸리지 않음")
                scale_right.markdown('<div class="scale-right">10 : 매우 심하게 졸림</div>',unsafe_allow_html=True)
        if st.form_submit_button("▥  상세 분석 시작하기 →",use_container_width=True):
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
            if d_sys is not None and d_dia is not None and d_sys<=d_dia:
                st.error(f"혈압을 다시 확인해 주세요. 수축기({d_sys})는 이완기({d_dia})보다 높아야 합니다.")
                st.stop()
            d_sleep=sleep_duration(d_bedtime,d_wake_time)
            save_and_analyze({"mode":"상세 폼","gender":d_gender,"age":d_age,"occupation":None,"height":d_height,"weight":d_weight,"bedtime":d_bedtime.strftime("%H:%M"),"wake_time":d_wake_time.strftime("%H:%M"),"sleep":d_sleep,"quality":None,"activity":d_activity,"stress":d_stress,"bmi":d_bmi,"sys":d_sys,"dia":d_dia,"heart_rate":d_hr,"daily_steps":d_steps,"sleep_disorder":"None","caffeine":d_caffeine,"recent_alcohol":d_recent_alcohol,"phone_hours":d_phone,"daytime_sleepiness":d_daytime_sleepiness,"night_awakenings":d_awakenings,"smoking":d_smoking})
else:
    d=st.session_state.data
    # 화면에 나가는 진단값은 전부 모델에서만 옵니다. 모델이 없거나 입력이 모자라면
    # 임의 수식으로 채우지 않고 안내를 띄웁니다.
    efficiency=predict_efficiency(d)
    verdict=predict_quality(d)
    lifestyle=predict_lifestyle_risk(d)
    st.markdown(f'<div class="eyebrow">YOUR SLEEP REPORT · {d["mode"]}</div><div class="hero">수면 건강 분석이<br><span class="blue">완료되었어요.</span></div><div class="sub">입력한 생활 습관을 기반으로 현재 수면 상태와 주요 위험 요인을 분석했습니다.</div>',unsafe_allow_html=True)

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
    status_metric(c,"흡연 여부",d["smoking"],d["smoking"],"bad" if d["smoking"]=="흡연" else "good")
    st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
    c4,c5=st.columns(2)
    status_metric(c4,"하루 카페인",f"{d['caffeine']:g}잔","섭취 조절 권장" if d['caffeine']>2 else "적정 수준","warn" if d['caffeine']>2 else "good")
    status_metric(c5,"최근 24시간 내 음주",d['recent_alcohol'],"수면 영향 가능" if d['recent_alcohol']=="음주함" else "음주 없음","warn" if d['recent_alcohol']=="음주함" else "good")
    with st.expander("입력한 상세 데이터 확인"):
        blood_pressure_text=f'{d["sys"]}/{d["dia"]} mmHg' if d["sys"] is not None and d["dia"] is not None else "미입력"
        heart_rate_text=f'{d["heart_rate"]}회/분' if d["heart_rate"] is not None else "미입력"
        daily_steps_text=f'{d["daily_steps"]:,}걸음' if d["daily_steps"] is not None else "미입력"
        details=[
            ("성별","여성" if d["gender"]=="Female" else "남성"),("나이",f'{d["age"]}세'),
            ("키",f'{d.get("height",0):g}cm'),("몸무게",f'{d.get("weight",0):g}kg'),
            ("BMI",f'{d["bmi"]:.1f}'),
            ("수면 패턴",f'{d["bedtime"]} ~ {d["wake_time"]}'),("자동 계산 수면시간",f'{d["sleep"]}시간'),
            ("밤중 깬 횟수",f'{d.get("night_awakenings","미입력")}회'),
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
