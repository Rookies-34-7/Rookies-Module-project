#  모델링

수면 건강 및 생활 습관 데이터를 바탕으로 **Quality of Sleep**을 예측하는 모델입니다.

---

## 1. 설명 
- **목적:** 수면 시간, 스트레스 지수, 신체 활동량, 혈압 등 생활 습관 요인을 분석하여 사용자의 수면 상태를 예측하는 모델 구축
- **타깃 변수 (Target):** `Quality of Sleep` (`High`, `Medium`, `Low`)
- **사용 모델:** `RandomForestClassifier` (랜덤 포레스트 분류기)
> **해석:** 75개 테스트 데이터 중 74개를 완벽히 분류하였으며, 단 1건의 오차(`Actual_Medium` → `Pred_High`)만 존재함.
---

##  2. 주요 성능 지표 
`class_weight='balanced'` 옵션을 적용하여 클래스 불균형을 완화한 후 평가한 모델 성능입니다.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Accuracy (정확도)** | **0.9733** | 
| **Precision (정밀도)** | **0.9744** |
| **Recall (재현율)** | **0.9733** |
| **F1-Score (F1점수)** | **0.9731** | 

## 혼동 행렬 (Confusion Matrix)
| Actual \ Predicted | Pred_High | Pred_Low | Pred_Medium |
| :--- | :---: | :---: | :---: |
| **Actual_High** | **36** | 0 | 0 |
| **Actual_Low** | 0 | **2** | 0 |
| **Actual_Medium** | 1 | 0 | **36** |

##  3. 기술 스택 및 개발 환경 (Tech Stack)
- **Language:** Python 3.12
- **Libraries:** `scikit-learn`, `pandas`, `joblib`
- **Environment:** VS Code

### 4.모델 성능 비교 (Random Forest vs XGBoost)

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | 0.9733 | 0.9744 | 0.9733 | 0.9731 | 1건 오분류 (`Medium` → `High`) |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0건 오분류 (전부 정답)** |

> **분석 요약:** XGBoosting 알고리즘 특성상 이전 트리의 오차를 순차적으로 보완하는 XGBoost가 경계선 부근의 미세한 패턴까지 오차 없이 100% 정확하게 분류했습니다.