# Sleep Efficiency Prediction Model

## 1. 모델 개요

사용자의 생활습관 및 수면 관련 정보를 기반으로 **Sleep Efficiency(수면 효율)**를 예측하는 회귀 모델입니다.

예측 결과는 Sleep Quality 모델의 결과와 함께 사용하여 사용자의 수면 상태 분석 및 생활습관 가이드에 활용합니다.

---

## 2. 데이터 및 전처리

전처리 담당자가 생성한 다음 파일을 사용했습니다.

`efficiency_prep_yt_ver.pkl`

### 입력 Feature

- Age
- Gender
- BMI
- Sleep_Duration
- Stress_Level
- Daytime_Sleepiness
- Caffeine_Intake_mg
- Physical_Activity_Minutes
- Screen_Time_Hours
- Smoking_Status
- Alcohol_Consumption

Target:

`Sleep_Efficiency`

숫자형 Feature에는 Scaling, 범주형 Feature에는 One-Hot Encoding을 적용하여 최종 **16개 Feature**를 모델에 입력합니다.

---

## 3. 모델 비교

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
| --- | ---: | ---: | ---: |
| Random Forest Baseline | 3.7183 | 4.6309 | 0.6426 |
| Random Forest Tuned | 3.6448 | 4.5217 | 0.6593 |
| Random Forest GridSearch | 3.6367 | 4.5117 | 0.6608 |
| **Linear Regression** | **3.6207** | **4.4964** | **0.6631** |
| Gradient Boosting | 3.6264 | 4.5024 | 0.6622 |

Random Forest Baseline에서 과적합이 확인되어 직접 하이퍼파라미터 튜닝을 수행했습니다.

추가로 **162개의 하이퍼파라미터 조합에 5-Fold 교차 검증을 적용하여 총 810회의 학습**을 수행했지만, Linear Regression이 가장 우수한 Test 성능을 보였습니다.

---

## 4. 최종 모델

최종 모델은 **Linear Regression**입니다.

- MAE: **3.6207**
- RMSE: **4.4964**
- R²: **0.6631**

Train/Test 성능 차이가 작고 비교한 모델 중 가장 좋은 Test 성능을 기록했으며, 모델이 단순하고 결과 해석이 용이하여 최종 모델로 선정했습니다.

최종 모델:

`sleep_efficiency_model.pkl`

---

## 5. 예측 테스트

저장된 모델과 전처리 객체를 불러와 새로운 사용자 입력에 대한 예측을 테스트했습니다.


=== Sleep Efficiency Prediction ===
예측 수면 효율: 84.83

모델의 전체적인 예측 경향은 확인되었으나, 매우 낮거나 높은 Sleep Efficiency에서는 예측 오차가 증가하는 경향이 있습니다.

---

## 6. 주요 Feature 분석

Linear Regression 계수 분석 결과 다음 Feature가 상대적으로 큰 영향을 보였습니다.

- `Stress_Level`
- `Daytime_Sleepiness`

특히 높은 `Stress_Level`과 `Daytime_Sleepiness`는 Sleep Efficiency 예측값을 낮추는 방향으로 나타났습니다.

단, 이는 모델이 데이터에서 학습한 통계적 관계이며 실제 인과관계를 의미하지 않습니다.

---

## 7. 사용 흐름

사용자 입력
    ↓
전처리
    ↓
Linear Regression
    ↓
Sleep Efficiency 예측
    ↓
Sleep Quality 결과와 종합
    ↓
사용자 수면 상태 분석 및 생활습관 가이드

