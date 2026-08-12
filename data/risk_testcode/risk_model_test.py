import joblib
import pandas as pd
import numpy as np

# 최종 모델 불러오기
model_package = joblib.load(
    "../../model/lifestyle_risk_model.pkl"
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
    "Screen_Time_Hours": 6.0,
    "Caffeine_Intake_mg": 150,
    "Stress_Level": "Medium",
    "Alcohol_Consumption": "No",
    "Physical_Activity_Minutes": 60.0,
    "Smoking_Status": "No"
}])


# 이상치 처리
for column, (lower, upper) in outlier_bounds.items():
    user_input[column] = user_input[column].clip(
        lower=lower,
        upper=upper
    )


# 숫자형 데이터 표준화
numeric_data = scaler.transform(
    user_input[numeric_features]
)


# 범주형 데이터 One-Hot Encoding
categorical_data = encoder.transform(
    user_input[categorical_features]
)


# 전처리된 데이터 결합
X_input = np.hstack([
    numeric_data,
    categorical_data
])


# 예측
prediction = model.predict(X_input)[0]


print("=== Lifestyle Risk Prediction ===")
print(f"예측 위험 지수: {prediction:.2f}")