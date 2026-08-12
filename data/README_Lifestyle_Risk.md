# Lifestyle Risk Prediction Model

생활습관 데이터를 기반으로 `Lifestyle_Risk_Index`를 예측하는 회귀 모델입니다.

## 1. 모델 입력값

- Age
- Gender
- BMI
- Sleep_Duration
- Screen_Time_Hours
- Caffeine_Intake_mg
- Stress_Level
- Alcohol_Consumption
- Physical_Activity_Minutes
- Smoking_Status

## 2. Target

`Lifestyle_Risk_Index`

생활습관 관련 정보를 종합한 위험 지수를 예측합니다.

## 3. 모델 비교

| Model | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: |
| Random Forest | 0.0408 | 0.0665 | 0.9974 |
| Linear Regression | 0.0138 | 0.0272 | 0.9996 |
| Gradient Boosting | 0.0472 | 0.0615 | 0.9978 |

세 모델을 동일한 데이터로 비교한 결과 Linear Regression이
가장 낮은 MAE와 RMSE, 가장 높은 R²를 기록하여 최종 모델로 선정했습니다.

## 4. 최종 모델

- Model: Linear Regression
- MAE: 0.0138
- RMSE: 0.0272
- R²: 0.9996


## 5. 전처리

전처리 담당자가 생성한 전처리 데이터를 사용했습니다.

`Sleep_preprocessing_yt_상관관계 반영ver.pkl`

해당 파일에는 다음 전처리가 적용된 Train/Test 데이터와 전처리 객체가 저장되어 있습니다.

- Train/Test 분리
- 이상치 처리
- 숫자형 Feature 표준화
- 범주형 Feature One-Hot Encoding

### 전처리 후 데이터

- X_train: (24000, 15)
- X_test: (6000, 15)
- y_train: (24000,)
- y_test: (6000,)

Target은 `Lifestyle_Risk_Index`이며, 원본 10개 입력 Feature는 전처리 후 15개 Feature로 변환됩니다.

최종 모델 파일 `lifestyle_risk_model.pkl`에는 새로운 사용자 입력에도 동일한 전처리를 적용할 수 있도록 다음 객체를 함께 저장했습니다.

- Linear Regression Model
- StandardScaler
- OneHotEncoder
- Numeric Feature 목록
- Categorical Feature 목록
- 이상치 처리 기준

## 6. 파일 구성

| File | Description |
| --- | --- |
| `lifestyle_risk_training.ipynb` | 모델 학습, 비교 및 최종 모델 선정 과정 |
| `lifestyle_risk_model.pkl` | 최종 Linear Regression 모델 및 전처리 객체 |
| `lifestyle_risk_test.py` | 저장된 모델을 이용한 예측 테스트 |

## 7. 모델 테스트

`lifestyle_risk_test.py`를 실행하여 저장된 모델의 예측 동작을 확인할 수 있습니다.

테스트 과정:

사용자 입력 → 이상치 처리 → 표준화 → One-Hot Encoding → 모델 예측 → Lifestyle_Risk_Index 출력