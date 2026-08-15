# 배포 가이드

앱은 **모델이 없어도 기동**합니다. 모델이 없으면 해당 지표만 안내 카드로 표시되고,
`model/` 에 pkl을 넣고 앱을 재시작하면 그때부터 값이 채워집니다.
즉 **먼저 배포하고 모델을 나중에 넣는 순서**가 가능합니다.

---

## 1. 모델 넣는 위치

`model_service.py` 가 아래 순서로 찾고, **처음 발견한 파일**을 씁니다.

| 지표 | 탐색 순서 |
|---|---|
| 수면의 질 | `model/quality_model_rf.pkl` → `model/quality_model_info.pkl` → `model/quality_model.pkl` → `data/quality_data/quality_model_info.pkl` |
| 수면 효율 | `model/efficiency_model.pkl` → `model/sleep_efficiency_model.pkl` → `data/risk_data/efficiency_sleep_efficiency_model.pkl` |
| 생활습관 위험도 | 모델 없이 산출식으로 계산 (pkl 불필요) |

**새 모델로 교체하려면** 위 표의 **가장 앞 경로**에 파일을 두면 됩니다.
경로를 바꾸고 싶으면 `model_service.py` 의 `MODEL_CANDIDATES` 에 경로를 추가하세요.

### pkl 저장 형식

아래 세 가지 중 아무 형태나 인식합니다. 학습 담당자가 저장 방식을 바꿔도 앱은 수정하지 않아도 됩니다.

```python
# (1) 전처리기 내장형 — 현재 수면의 질 모델
joblib.dump({"model": clf, "preprocessor": column_transformer, "labels": list(clf.classes_)}, path)

# (2) 전처리 분리형 — 현재 수면 효율 모델
joblib.dump({"model": reg, "scaler": scaler, "encoder": encoder,
             "numeric_features": num_cols, "categorical_features": cat_cols}, path)

# (3) 추정기 단독 (feature_names_in_ 이 있어야 함)
joblib.dump(clf, path)
```

> 컬럼명은 공백/언더바를 가리지 않습니다 (`Sleep Duration` = `Sleep_Duration`).
> 다만 모델이 요구하는 컬럼이 폼에서 채울 수 없는 것이면 예측이 비활성화됩니다.
> 확인 방법은 아래 4번.

---

## 2. Streamlit Community Cloud 배포

1. GitHub 저장소를 연결하고 아래 값으로 앱을 만듭니다.

   | 항목 | 값 |
   |---|---|
   | Main file path | `streamlit-ui/streamlit_app.py` |
   | Python version | **3.11 이상** (로컬과 동일한 3.12 권장) |

   > `scikit-learn>=1.9` 가 Python 3.11+ 를 요구합니다. 3.10 이하를 고르면 설치가 실패합니다.

2. **Advanced settings → Secrets** 에 아래를 붙여 넣습니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   # OPENAI_MODEL = "gpt-4.1-mini"   # 생략 시 기본값
   ```

   `.env` 는 `.gitignore` 대상이라 배포 서버에 올라가지 않습니다.
   앱은 **secrets → 환경변수** 순으로 키를 찾으므로, 로컬은 `.env`, 배포는 Secrets를 쓰면 됩니다.
   키를 넣지 않아도 앱은 뜨고, AI 코치만 안내 문구를 표시합니다.

3. 의존성은 `requirements.txt` 를 씁니다. 저장소 루트와 `streamlit-ui/` 양쪽에 있으니
   **둘 중 하나를 고치면 나머지도 같이 맞춰 주세요.**

---

## 3. 저장소에서 지우면 안 되는 것

학습용 자료처럼 보이지만 **런타임에 읽는 파일**이 있습니다. 지우면 화면이 깨집니다.

| 파일 | 용도 |
|---|---|
| `data/risk_data/Sleep Health and Lifestyle Dataset.csv` | 레이더 차트의 백분위 기준 분포 |
| `data/quality_data/Sleep_health_and_lifestyle_dataset.csv` | 폼에 없는 범주형의 주변화 가중치 |
| `streamlit-ui/assets/` | 배경 이미지, 배경음악 |

반대로 `data/*_testcode/` 와 `data/risk_data/risk_prep_*.pkl`(약 11MB)은
학습 단계 산출물이라 실행에는 필요 없습니다. 저장소를 가볍게 하려면 정리 대상입니다.

---

## 4. 모델이 안 붙을 때

로컬에서 아래를 실행하면 **어떤 pkl이 잡혔는지, 어떤 컬럼을 못 채우는지** 바로 나옵니다.

```bash
python streamlit-ui/model_service.py
```

출력 예:

```
[quality] 로드 성공: model\quality_model_rf.pkl
  요구 컬럼 10개:
    - Gender                       폼에서 채움
    ...
  샘플 예측: {'label': 'High', 'text': '양호', 'score': 8, ...}
```

- `모델 없음` → 경로 확인 (1번 표)
- `!! 채울 수 없음` → 해당 컬럼을 `FIELD_SOURCES` 에 매핑 추가
- `주변화 대상이 N개` → 현재 구현은 1개만 주변화하므로 나머지는 폼에서 받아야 함

> 화면에 뜨는 "입력이 모두 채워졌는지 확인해 주세요" 안내는 **입력 부족과 모델 미탑재를
> 구분하지 않습니다.** 배포 후 값이 안 나오면 입력부터 의심하지 말고 위 명령으로 확인하세요.

---

## 5. 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit-ui/streamlit_app.py
```
