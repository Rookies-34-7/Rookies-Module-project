#  모델링 (작성자: 장은빈)

수면 건강 및 생활 습관 데이터를 바탕으로 **Quality of Sleep**을 예측하는 모델입니다.

---

## 1. 프로젝트 개요 & 설명
- **목적:** 수면 시간, 스트레스 지수, 신체 활동량, 혈압 등 주요 생활 습관 요인을 분석하여 수면 상태 예측
- **전처리 최적화:** 데이터 노이즈를 유발하던 `Occupation`(직업) 변수를 제거하여 핵심 피처 중심의 최적화 진행
- **타깃 변수 (Target):** `Quality of Sleep` (`High`, `Medium`, `Low`)
- **사용 모델:** `RandomForestClassifier` (랜덤 포레스트 분류기)
> **해석:** 피처 최적화 후 75개의 테스트 데이터 전체를 오차 없이 100% 완벽하게 분류함.

---

## 2. 주요 성능 지표 & 혼동 행렬
`class_weight='balanced'` 옵션을 적용하여 클래스 불균형을 완화한 후 평가한 Random Forest 모델의 성능입니다.

###  Performance Metrics
| Metric | Score |
| :--- | :---: | 
| **Accuracy (정확도)** | **1.0000** | 
| **Precision (정밀도)** | **1.0000** |
| **Recall (재현율)** | **1.0000** |
| **F1-Score (F1점수)** | **1.0000** | 

###  Confusion Matrix
| Actual \ Predicted | Pred_High | Pred_Low | Pred_Medium |
| :--- | :---: | :---: | :---: |
| **Actual_High** | **36** | 0 | 0 |
| **Actual_Low** | 0 | **2** | 0 |
| **Actual_Medium** | 0 | 0 | **37** |

---

## 3. 모델 성능 비교 (Random Forest vs XGBoost)

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0건 오분류 (전부 정답)** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0건 오분류 (전부 정답)** |

> **분석 요약:** 다중 범주형 변수인 `Occupation`을 제거함에 따라 모델이 수면 상태 판단에 본질적인 피처들에 집중할 수 있게 되었으며, 그 결과 Random Forest와 XGBoost 두 모델 모두 100% 분류 정확도를 달성하였습니다. 이에 따라 상대적으로 연산 구조가 단순하고 일반화 능력이 우수한 **Random Forest**를 최종 기본 운용 모델로 확정하였습니다.

---

## 4. 기술 스택 및 개발 환경 (Tech Stack)
- **Language:** Python 3.12
- **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`
- **Environment:** VS Code