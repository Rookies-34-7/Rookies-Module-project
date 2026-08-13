"""학습된 모델 추론을 화면 코드와 분리해 담당합니다.

설계 원칙: **모델 파일만 갈아끼우면 바로 동작해야 한다.**
그래서 필요한 입력 컬럼을 코드에 박지 않고 전처리기에서 직접 읽어옵니다.
새 모델의 피처가 늘거나 줄어도, 아래 FIELD_SOURCES에 그 항목이 있으면 그대로 돌아갑니다.

── 새 모델 받았을 때 하는 일 ──────────────────────────────
  1. pkl을 model/ 폴더에 넣습니다. 파일명은 아래 MODEL_CANDIDATES 참고.
  2. 앱을 재시작합니다. (@st.cache_resource라 재시작 전엔 옛 모델이 남습니다)
  3. 잘 붙었는지 보려면:  python model_service.py
     -> 어떤 컬럼을 요구하는지, 폼에서 못 채우는 게 있는지 출력합니다.

지원하는 pkl 형식
  * {"model":…, "preprocessor":…, "labels":…}   <- 현재 quality 모델
  * {"model":…, "scaler":…, "encoder":…, …}     <- 현재 risk 모델
  * 전처리기 없는 추정기 단독

현재 연결
  * quality : 수면의 질 3분류 -> 양호/주의/위험
추가 예정
  * 효율 모델(Sleep_Efficiency) -> 수면 건강 점수 게이지
  * 신규 risk 모델(수면 장애)    -> 수면 장애 위험도 막대
"""
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_resource = st.cache_resource(show_spinner=False)
except Exception:                      # 터미널에서 직접 실행할 때(자가진단)
    def cache_resource(fn):
        return fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 앞에 있는 경로부터 찾습니다. 팀원이 준 파일을 model/ 에 넣으면 그쪽이 우선합니다.
# 세 모델 모두 교체 예정이라, 파일이 없으면 조용히 None을 돌려 화면이 기존 로직으로 동작합니다.
MODEL_CANDIDATES = {
    # 수면의 질 3분류 -> 양호/주의/위험
    "quality": [
        PROJECT_ROOT / "model" / "quality_model_rf.pkl",
        PROJECT_ROOT / "model" / "quality_model_info.pkl",
        PROJECT_ROOT / "model" / "quality_model.pkl",
        PROJECT_ROOT / "data" / "quality_data" / "quality_model_info.pkl",
    ],
    # 수면 효율(Sleep_Efficiency) 회귀 -> 수면 건강 점수 게이지
    "efficiency": [
        PROJECT_ROOT / "model" / "efficiency_model.pkl",
        PROJECT_ROOT / "model" / "sleep_efficiency_model.pkl",
    ],
    # 수면 장애 분류 -> 수면 장애 위험도 막대
    "risk": [
        PROJECT_ROOT / "model" / "risk_model_new.pkl",
        PROJECT_ROOT / "model" / "sleep_disorder_model.pkl",
        PROJECT_ROOT / "model" / "risk_model.pkl",
    ],
}

# 수면 효율 게이지 눈금. 임상 기준 85% 이상 정상, 75% 미만 주의.
# 학습 데이터 실측 범위가 57~99라 0부터 그리면 바늘이 항상 오른쪽에 몰립니다.
EFFICIENCY_GAUGE = {"min": 55, "max": 100, "warn": 75, "good": 85}
QUALITY_CSV = PROJECT_ROOT / "data" / "quality_data" / "Sleep_health_and_lifestyle_dataset.csv"

# 예측 라벨 -> (화면 문구, 배지 색, 아이콘)
QUALITY_LABELS = {
    "High":   ("양호", "good", "✅"),
    "Medium": ("주의", "warn", "⚠️"),
    "Low":    ("위험", "bad",  "🚨"),
}

# 커피 한 잔의 카페인. 폼은 '잔'으로 받고 모델은 mg으로 학습돼 있습니다.
MG_PER_CUP = 95


def _bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _norm(name):
    """'Sleep Duration'과 'Sleep_Duration'을 같은 키로 취급합니다."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# 모델이 요구하는 컬럼(정규화된 이름) -> 폼 dict에서 값을 꺼내는 함수.
# 새 모델이 다른 피처를 쓰더라도 여기에 있으면 코드 수정 없이 채워집니다.
FIELD_SOURCES = {
    "age":                     lambda d: d.get("age"),
    "gender":                  lambda d: d.get("gender"),
    "bmi":                     lambda d: d.get("bmi"),
    "bmicategory":             lambda d: _bmi_category(d["bmi"]) if d.get("bmi") is not None else None,
    "sleepduration":           lambda d: d.get("sleep"),
    "physicalactivitylevel":   lambda d: d.get("activity"),
    "physicalactivityminutes": lambda d: d.get("activity"),
    "stresslevel":             lambda d: d.get("stress"),
    "heartrate":               lambda d: d.get("heart_rate"),
    "dailysteps":              lambda d: d.get("daily_steps"),
    "systolicbp":              lambda d: d.get("sys"),
    "diastolicbp":             lambda d: d.get("dia"),
    "screentimehours":         lambda d: d.get("phone_hours"),
    "daytimesleepiness":       lambda d: d.get("daytime_sleepiness"),
    "caffeineintakemg":        lambda d: None if d.get("caffeine") is None else d["caffeine"] * MG_PER_CUP,
    "smokingstatus":           lambda d: {"흡연": "Yes", "비흡연": "No"}.get(d.get("smoking")),
    "alcoholconsumption":      lambda d: {"음주함": "Yes", "음주 안 함": "No"}.get(d.get("recent_alcohol")),
}

# 전처리기가 학습한 BMI 범주는 Normal/Obese/Overweight 3종뿐입니다. 학습에 없는 값을
# 그대로 넣으면 원핫이 전부 0인, 한 번도 본 적 없는 패턴이 되므로 Normal로 모읍니다.
CATEGORY_ALIAS = {
    "bmicategory": {"Underweight": "Normal", "Normal Weight": "Normal"},
}


@cache_resource
def load_bundle(kind="quality"):
    """pkl을 읽어 (estimator, preprocessor, labels)로 정규화합니다. 실패하면 None."""
    try:
        import joblib
    except ImportError:
        return None

    for path in MODEL_CANDIDATES.get(kind, []):
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
        except Exception:
            continue

        bundle = {"path": path, "pre": None, "scaler": None, "encoder": None,
                  "numeric": [], "categorical": []}
        if isinstance(obj, dict):
            estimator = obj.get("model")
            bundle["pre"] = obj.get("preprocessor")
            # scaler/encoder를 따로 담은 형식(현 risk 모델)도 지원합니다.
            bundle["scaler"] = obj.get("scaler")
            bundle["encoder"] = obj.get("encoder")
            bundle["numeric"] = list(obj.get("numeric_features", []))
            bundle["categorical"] = list(obj.get("categorical_features", []))
            labels = list(obj.get("labels", getattr(estimator, "classes_", [])))
        else:
            estimator, labels = obj, list(getattr(obj, "classes_", []))

        if estimator is not None:
            bundle["estimator"] = estimator
            bundle["labels"] = labels
            return bundle
    return None


def required_columns(bundle):
    """모델이 요구하는 입력 컬럼을 번들에서 뽑아냅니다.

    ColumnTransformer면 feature_names_in_에서, scaler+encoder 분리형이면
    numeric_features/categorical_features 목록에서 가져옵니다.
    """
    if bundle.get("numeric") or bundle.get("categorical"):
        return list(bundle["numeric"]) + list(bundle["categorical"])
    for holder in (bundle.get("pre"), bundle.get("estimator")):
        names = getattr(holder, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return []


def _categorical_choices(bundle):
    """인코더가 학습한 범주 목록을 {컬럼: [값…]}로 돌려줍니다."""
    out = {}
    for _, trans, cols in getattr(bundle.get("pre"), "transformers_", []):
        cats = getattr(trans, "categories_", None)
        if cats is None or isinstance(cols, str):
            continue
        for col, values in zip(cols, cats):
            out[col] = list(values)
    encoder = bundle.get("encoder")
    if encoder is not None and bundle.get("categorical"):
        for col, values in zip(bundle["categorical"], getattr(encoder, "categories_", [])):
            out[col] = list(values)
    return out


def _transform(bundle, frame):
    """번들 형식에 맞게 전처리를 적용합니다."""
    if bundle.get("pre") is not None:
        return bundle["pre"].transform(frame)
    if bundle.get("scaler") is not None and bundle.get("encoder") is not None:
        X_num = bundle["scaler"].transform(frame[bundle["numeric"]])
        X_cat = bundle["encoder"].transform(frame[bundle["categorical"]])
        if hasattr(X_cat, "toarray"):
            X_cat = X_cat.toarray()
        return np.hstack([X_num, X_cat])
    return frame


# 스트레스는 모델마다 표현이 다릅니다. quality는 숫자(학습 분포 3~8),
# risk는 Low/Medium/High 범주(원본 데이터에 4/7/9 세 값만 존재).
# 폼은 1~10 슬라이더를 유지하고, 범주를 요구하는 모델에만 아래 기준으로 변환합니다.
# ※ 경계값은 팀 확인이 필요합니다.
STRESS_BANDS = ((4, "Low"), (7, "Medium"), (10, "High"))


def _to_stress_band(value):
    for upper, label in STRESS_BANDS:
        if value <= upper:
            return label
    return STRESS_BANDS[-1][1]


def _build_row(d, columns, choices=None):
    """폼 dict에서 모델이 요구하는 컬럼만 채웁니다. 못 채운 건 따로 알려줍니다.

    choices를 주면, 범주형 컬럼에 숫자가 들어가는 경우를 잡아 변환합니다.
    (예: 스트레스를 1~10으로 받았는데 모델은 Low/Medium/High를 기대)
    """
    choices = choices or {}
    row, missing = {}, []
    for col in columns:
        source = FIELD_SOURCES.get(_norm(col))
        value = source(d) if source else None
        if value is None:
            missing.append(col)

        allowed = choices.get(col)
        if allowed and isinstance(value, (int, float)):
            if _norm(col) == "stresslevel":
                value = _to_stress_band(value)
            else:
                # 숫자를 그대로 넣으면 원핫이 전부 0이 되어 조용히 왜곡됩니다.
                value = None
                missing.append(col)

        alias = CATEGORY_ALIAS.get(_norm(col), {})
        row[col] = alias.get(value, value)
    return row, missing


def predict_quality(d):
    """폼 입력으로 수면의 질을 예측합니다.

    필수 항목이 비었으면 None을 돌려 호출부가 기존 규칙으로 폴백하게 합니다.
    빈 값을 평균으로 채우지 않는 이유: 랜덤포레스트는 평균값이 '중립'이 아니라
    특정 분기 경로로 조용히 밀려서 근거 없는 판정이 나옵니다.
    """
    bundle = load_bundle("quality")
    if bundle is None:
        return None

    X, weights = _predict_frame(bundle, d)
    if X is None:
        return None

    try:
        averaged = np.average(bundle["estimator"].predict_proba(X), axis=0, weights=weights)
    except Exception:
        return None

    columns = required_columns(bundle)
    choices = _categorical_choices(bundle)
    _, missing = _build_row(d, columns, choices)
    marginal_col = next((c for c in missing if c in choices), None)
    labels = bundle["labels"] or list(bundle["estimator"].classes_)
    label = str(labels[int(averaged.argmax())])
    text, tone, icon = QUALITY_LABELS.get(label, (label, "warn", "⚠️"))
    return {
        "label": label,
        "text": text,
        "tone": tone,
        "icon": icon,
        "proba": {str(l): float(p) for l, p in zip(labels, averaged)},
        "marginalized": marginal_col,
    }


def _predict_frame(bundle, d):
    """폼 dict를 모델 입력으로 바꿔 전처리까지 끝낸 행렬과 주변화 가중치를 돌려줍니다.

    predict_quality / predict_efficiency / predict_risk가 공통으로 씁니다.
    채울 수 없는 컬럼이 있으면 (None, None)을 돌려 호출부가 폴백하게 합니다.
    """
    columns = required_columns(bundle)
    if not columns:
        return None, None

    choices = _categorical_choices(bundle)
    row, missing = _build_row(d, columns, choices)

    # 폼에서 아예 받지 않는 범주형(예: 직업)은 학습된 값 전체로 주변화합니다.
    marginal_col = next((c for c in missing if c in choices), None)
    if [c for c in missing if c != marginal_col]:
        return None, None

    if marginal_col:
        values = choices[marginal_col]
        frame = pd.DataFrame([row] * len(values))
        frame[marginal_col] = values
        weights = _marginal_weights(marginal_col, tuple(values))
    else:
        frame, weights = pd.DataFrame([row]), None

    try:
        X = _transform(bundle, frame)
    except Exception:
        return None, None
    return X, weights


def predict_efficiency(d):
    """수면 효율(%)을 예측합니다. 모델이 없으면 None.

    회귀 모델이라 predict()를 그대로 쓰고, 게이지 구간은 EFFICIENCY_GAUGE를 따릅니다.
    """
    bundle = load_bundle("efficiency")
    if bundle is None:
        return None
    X, weights = _predict_frame(bundle, d)
    if X is None:
        return None
    try:
        pred = bundle["estimator"].predict(X)
        value = float(np.average(pred, weights=weights))
    except Exception:
        return None

    g = EFFICIENCY_GAUGE
    tone = "good" if value >= g["good"] else ("warn" if value >= g["warn"] else "bad")
    return {"value": value, "tone": tone, "gauge": g}


def predict_risk(d):
    """수면 장애별 확률(%)을 예측합니다. 모델이 없으면 None.

    분류 모델의 predict_proba를 그대로 막대그래프에 씁니다.
    반환 예: {"불면증": 32.1, "수면무호흡증": 21.4, ...}
    """
    bundle = load_bundle("risk")
    if bundle is None:
        return None
    estimator = bundle["estimator"]
    if not hasattr(estimator, "predict_proba"):
        return None          # 회귀 모델이면 막대그래프에 못 씁니다.
    X, weights = _predict_frame(bundle, d)
    if X is None:
        return None
    try:
        proba = np.average(estimator.predict_proba(X), axis=0, weights=weights)
    except Exception:
        return None
    labels = bundle["labels"] or list(estimator.classes_)
    return {str(l): float(p) * 100 for l, p in zip(labels, proba)}


@cache_resource
def _marginal_weights(column, values):
    """주변화 가중치. 학습 데이터의 실제 분포를 쓰고, 없으면 균등."""
    try:
        counts = pd.read_csv(QUALITY_CSV)[column].value_counts(normalize=True)
        w = counts.reindex(list(values)).fillna(0).to_numpy(dtype=float)
        if w.sum() > 0:
            return w / w.sum()
    except Exception:
        pass
    return np.full(len(values), 1 / len(values))


SAMPLE_INPUT = {
    "age": 32, "gender": "Female", "bmi": 22.5, "sleep": 7.0, "activity": 45,
    "stress": 5, "heart_rate": 70, "daily_steps": 7000, "sys": 120, "dia": 80,
    "phone_hours": 5.0, "daytime_sleepiness": 5, "caffeine": 1.0,
    "smoking": "비흡연", "recent_alcohol": "음주 안 함",
}


def diagnose():
    """`python model_service.py`로 실행하면 세 모델이 잘 붙었는지 점검합니다.

    새 pkl을 model/ 에 넣은 뒤 이걸 먼저 돌려보면, 폼에서 못 채우는 컬럼이 있는지
    앱을 띄우기 전에 알 수 있습니다.
    """
    for kind, predict in (("quality", predict_quality),
                          ("efficiency", predict_efficiency),
                          ("risk", predict_risk)):
        print("=" * 72)
        bundle = load_bundle(kind)
        if bundle is None:
            print(f"[{kind}] 모델 없음 — 화면은 기존 로직으로 동작합니다.")
            print("  아래 경로 중 하나에 pkl을 두면 자동으로 잡힙니다:")
            for p in MODEL_CANDIDATES[kind]:
                print("   -", p.relative_to(PROJECT_ROOT))
            continue

        columns = required_columns(bundle)
        choices = _categorical_choices(bundle)
        print(f"[{kind}] 로드 성공: {bundle['path'].relative_to(PROJECT_ROOT)}")
        print(f"  추정기: {type(bundle['estimator']).__name__}   라벨: {bundle['labels'] or '(회귀)'}")
        print(f"  요구 컬럼 {len(columns)}개:")
        for c in columns:
            if _norm(c) in FIELD_SOURCES:
                src = "폼에서 채움"
            elif c in choices:
                src = f"폼에 없음 -> 학습 분포로 주변화 (범주 {len(choices[c])}종)"
            else:
                src = "!! 채울 수 없음 — FIELD_SOURCES에 매핑을 추가하세요"
            print(f"    - {c:28s} {src}")

        unfillable = [c for c in columns if _norm(c) not in FIELD_SOURCES and c not in choices]
        marginal = [c for c in columns if _norm(c) not in FIELD_SOURCES and c in choices]
        if len(marginal) > 1:
            print(f"\n  주의: 주변화 대상이 {len(marginal)}개입니다({', '.join(marginal)}). "
                  "현재 구현은 1개만 주변화하므로 나머지는 폼에서 받아야 합니다.")
        if unfillable:
            print(f"\n  ** 이 컬럼들 때문에 예측이 안 됩니다: {', '.join(unfillable)}")

        print(f"\n  샘플 예측: {predict(SAMPLE_INPUT)}")


if __name__ == "__main__":
    diagnose()
