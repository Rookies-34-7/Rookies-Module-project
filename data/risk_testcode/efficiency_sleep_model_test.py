import joblib
import pandas as pd
import numpy as np


# 최종 Sleep Efficiency 모델 불러오기
model_package = joblib.load(
    #"../risk_data/sleep_efficiency_model.pkl"
    "C:\\Users\\EZ\\Desktop\\루키즈\\모듈프로젝트(1)\\Rookies-Module-project\\data\\risk_data\\efficiency_sleep_efficiency_model.pkl"
)

model = model_package["model"]
scaler = model_package["scaler"]
encoder = model_package["encoder"]
numeric_features = model_package["numeric_features"]
categorical_features = model_package["categorical_features"]
outlier_bounds = model_package["outlier_bounds"]


# 테스트용 사용자 입력값
user_input = pd.DataFrame([{
    "Age": 25,
    "Gender": "Male",
    "BMI": 23.0,
    "Sleep_Duration": 7.0,
    "Daytime_Sleepiness": 3,
    "Caffeine_Intake_mg": 150,
    "Physical_Activity_Minutes": 60.0,
    "Screen_Time_Hours": 5.0,
    "Stress_Level": "Medium",
    "Smoking_Status": "No",
    "Alcohol_Consumption": "No"
}])


# 학습 데이터와 동일하게 이상치 처리
for column, (lower, upper) in outlier_bounds.items():
    user_input[column] = user_input[column].clip(
        lower=lower,
        upper=upper
    )


# 숫자형 Feature 표준화
numeric_data = scaler.transform(
    user_input[numeric_features]
)


# 범주형 Feature One-Hot Encoding
categorical_data = encoder.transform(
    user_input[categorical_features]
)


# 전처리 결과 결합
X_input = np.hstack([
    numeric_data,
    categorical_data
])


# Sleep Efficiency 예측
prediction = model.predict(X_input)[0]

print("=== Sleep Efficiency Prediction ===")
print(f"예측 수면 효율: {prediction:.2f}")