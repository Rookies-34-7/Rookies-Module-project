# SomniAI Streamlit 버전

## 실행 방법

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`을 열면 됩니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `streamlit_app.py` | 앱 본체 (입력 폼 + 결과 리포트) |
| `style.css` | 전체 스타일. `load_css()`가 읽어서 주입합니다 |
| `.streamlit/config.toml` | 기본 테마(색상) 설정 |
| `requirements.txt` | 의존성 (streamlit, plotly) |

> `style.css`만 수정하면 Streamlit이 변경을 감지하지 못합니다. 파일 감시자가 `.py`만 보기 때문에,
> CSS만 고쳤을 때는 브라우저에서 `R`을 누르거나 새로고침해야 반영됩니다.

## 포함 기능

- 수면 건강 데이터 입력 폼 (간단 측정 / 상세 측정)
- 불면증·수면무호흡증·수면 부족 위험도 그래프
- 수면 건강 점수 게이지와 양호/위험 판정
- 주요 지표 카드와 개인 맞춤 권고
- 팝업형 AI 수면 코치 챗봇 UI

## 결과값은 현재 더미입니다

위험도·점수·판정·카드 상태 배지는 **전부 고정 더미값**입니다. 입력값을 바꿔도 변하지 않습니다.
카드에 표시되는 **값 자체는** 사용자가 입력한 값을 그대로 보여줍니다.

실제 모델을 붙일 때는 `streamlit_app.py` 상단의 더미 블록만 교체하면 되고, 화면 코드는 손댈 필요가 없습니다.

```python
DUMMY_SCORES  = {"insomnia": 37, "apnea": 31, "shortage": 44, "quality": 63}
DUMMY_VERDICT = "good"          # "good" → 양호,  "bad" → 위험
DUMMY_STATUS  = { ... }         # 카드별 (상태 문구, 색상 톤)
DUMMY_CHAT_ANSWER = "..."       # 챗봇 응답
```

배지 3색(good/warn/bad)이 한 화면에 모두 보이도록 `DUMMY_STATUS`의 상태를 섞어 두었습니다.
`DUMMY_VERDICT`를 `"bad"`로 바꾸면 위험 판정 화면을 확인할 수 있습니다.

입력값을 그대로 보여주기 위한 변환 두 가지(`sleep_duration()`으로 취침·기상 시각에서 수면시간 계산,
키·몸무게에서 BMI 계산)는 예측이 아니라서 그대로 남아 있습니다.
