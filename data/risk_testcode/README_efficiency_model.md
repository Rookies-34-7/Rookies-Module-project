# Sleep Efficiency Prediction Model

## 1. 모델 개요

사용자의 생활습관 및 수면 관련 정보를 기반으로 **Sleep Efficiency(수면 효율)**를 예측하는 회귀 모델입니다.

예측 결과는 Sleep Quality 모델의 결과와 함께 사용하여 사용자의 수면 상태 분석 및 생활습관 가이드에 활용합니다.

---

## 2. 데이터 및 전처리

최종 모델은 총 **11개의 사용자 입력 Feature**를 사용합니다.

### 입력 Feature

- Age
- Gender
- BMI
- Sleep_Duration
- Daytime_Sleepiness
- Caffeine_Intake_mg
- Physical_Activity_Minutes
- Screen_Time_Hours
- Night_Awakenings
- Smoking_Status
- Alcohol_Consumption

Target:

`Sleep_Efficiency`

초기 모델에서 사용했던 `Stress_Level`은 Sleep Duration의 영향이 모델에 충분히 반영되지 않는 현상을 분석한 결과 최종 입력 Feature에서 제외했습니다.

대신 수면 중 각성 정보를 나타내는 `Night_Awakenings`를 추가하여 최종 입력 Feature 수는 기존과 동일한 11개로 유지했습니다.

### 전처리

숫자형 Feature에는 `StandardScaler`를 적용하고 범주형 Feature에는 `One-Hot Encoding`을 적용했습니다.

- 숫자형 Feature: 8개
- 범주형 Feature: 3개
- One-Hot Encoding 이후 최종 모델 입력: **14개 Feature**

일부 Feature에는 기존 전처리 기준과 동일한 값 범위 제한을 적용했습니다.

- `Caffeine_Intake_mg`: 0 ~ 395
- `Physical_Activity_Minutes`: 0 ~ 104.4

---

## 3. 기존 모델 분석

초기 모델에서는 다음 알고리즘을 비교했습니다.

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
| --- | ---: | ---: | ---: |
| Random Forest Baseline | 3.7183 | 4.6309 | 0.6426 |
| Random Forest Tuned | 3.6448 | 4.5217 | 0.6593 |
| Random Forest GridSearch | 3.6367 | 4.5117 | 0.6608 |
| Linear Regression | 3.6207 | 4.4964 | 0.6631 |
| Gradient Boosting | 3.6264 | 4.5024 | 0.6622 |

초기 모델에서는 Linear Regression이 가장 높은 Test 성능을 기록했습니다.

하지만 사용자 입력값 중 `Sleep_Duration`을 변화시켜 예측값을 확인한 결과, 수면시간이 크게 변해도 Sleep Efficiency 예측값이 거의 변하지 않는 문제가 확인되었습니다.

---

## 4. Sleep Duration 영향도 분석

Sleep Duration의 영향이 작게 나타나는 원인을 확인하기 위해 Feature 간 관계와 다양한 모델을 추가로 분석했습니다.

전체 데이터에서는 `Sleep_Duration`과 `Sleep_Efficiency` 사이에 양의 상관관계가 나타났지만, `Stress_Level`을 동일하게 통제한 경우 해당 관계가 크게 감소했습니다.

또한 다음 방법을 추가로 실험했습니다.

- Random Forest
- Linear Regression
- Gradient Boosting
- Deep Learning
- Night_Awakenings 추가
- Sleep Duration 관련 파생변수 생성
- Feature Interaction 추가

여러 모델과 파생변수를 적용해도 비슷한 현상이 반복되어 특정 알고리즘의 문제가 아닌 데이터 Feature 간 관계의 영향으로 판단했습니다.

특히 `Stress_Level`을 제거했을 때 Sleep Duration 변화에 대한 모델의 민감도가 크게 증가하는 것을 확인했습니다.

---

## 5. Stress Level 제외 모델 비교

`Stress_Level`을 제거하고 `Night_Awakenings`를 추가한 11개의 입력 Feature를 기준으로 모델을 다시 학습했습니다.

| Model | Train MAE | Test MAE | Test RMSE | Train R² | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 1.4295 | 3.8463 | 4.7983 | 0.9462 | 0.6163 |
| Linear Regression | 3.7904 | 3.8039 | 4.7442 | 0.6265 | 0.6249 |
| **Gradient Boosting** | **3.7049** | **3.7659** | **4.6931** | **0.6436** | **0.6329** |

Random Forest는 Train과 Test 성능 차이가 크게 나타나 과적합 경향을 보였습니다.

Gradient Boosting은 세 모델 중 가장 좋은 Test 성능을 기록했으며 Train/Test 성능 차이도 비교적 작게 나타났습니다.

---

## 6. Sleep Duration 민감도 테스트

다른 사용자 입력값을 동일하게 유지하고 `Sleep_Duration`만 변경하여 최종 Gradient Boosting 모델의 예측값 변화를 확인했습니다.

| Sleep Duration | Predicted Sleep Efficiency |
| ---: | ---: |
| 2시간 | 79.56 |
| 4시간 | 79.76 |
| 6시간 | 81.82 |
| 7시간 | 83.59 |
| 8시간 | 84.34 |
| 9시간 | 84.50 |
| 10시간 | 82.94 |

기존 모델과 달리 Sleep Duration 변화에 따라 예측값이 뚜렷하게 변화했습니다.

특히 7~9시간 구간까지 예측값이 증가한 뒤 10시간에서 감소하는 형태가 나타났습니다.

---

## 7. Night Awakenings 분석

`Night_Awakenings`를 0회부터 7회까지 변경하여 민감도를 확인했습니다.

Gradient Boosting 모델에서는 Night Awakenings 변화에 따른 Sleep Efficiency 예측값 변화가 크지 않았습니다.

따라서 `Night_Awakenings` 추가 자체가 모델 성능을 크게 개선한 것은 아니며, 이번 모델 변화에서 가장 큰 영향을 준 요소는 `Stress_Level` 제거로 판단했습니다.

---

## 8. 최종 모델

최종 모델은 **Gradient Boosting Regressor**입니다.

- Train MAE: **3.7049**
- Test MAE: **3.7659**
- Test RMSE: **4.6931**
- Train R²: **0.6436**
- Test R²: **0.6329**

초기 Linear Regression 모델보다 Test 성능은 일부 감소했지만, 서비스의 주요 사용자 입력값인 `Sleep_Duration` 변화가 예측 결과에 보다 명확하게 반영되는 것을 확인했습니다.

이에 따라 단순히 Test R²가 가장 높은 모델을 선택하기보다, 예측 성능과 사용자 입력 변화에 대한 모델의 민감도를 함께 고려하여 Gradient Boosting을 최종 모델로 선정했습니다.

최종 모델:

`efficiency_sleep_efficiency_model.pkl`

PKL 파일에는 다음 객체가 함께 저장됩니다.

- Gradient Boosting 모델
- StandardScaler
- OneHotEncoder
- Numeric Feature 목록
- Categorical Feature 목록
- 전체 입력 Feature 목록
- 이상치 처리 기준
- Target 정보
- 모델 성능 지표

---

## 9. 사용 흐름

사용자 입력
    ↓
전처리
    ↓
Gradient Boosting
    ↓
Sleep Efficiency 예측
    ↓
Sleep Quality 결과와 종합
    ↓
사용자 수면 상태 분석 및 생활습관 가이드