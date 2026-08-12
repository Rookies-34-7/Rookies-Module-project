import joblib
import pandas as pd
import sklearn
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix )

# 1. 데이터 로드
df = joblib.load('sleep_preprocessed.pkl')

# 2. 저장된 변수들 추출
X_train = df['X_train']
X_test = df['X_test']
y_train = df['y_train']
y_test = df['y_test']

# 3. 타겟 변수 Label Encoding 
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 4. XGBoost 모델 생성 및 학습
model_xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)
model_xgb.fit(X_train, y_train_encoded)

print('=== XGBoost 학습 완료 ===\n')

# 5. 예측 및 평가
y_pred_encoded = model_xgb.predict(X_test)

# 숫자 -> 문자 변환
y_pred = le.inverse_transform(y_pred_encoded)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=== XGBoost 핵심 성능 지표 ===")
print(f"1. Accuracy  (정확도) : {acc:.4f}")
print(f"2. Precision (정밀도) : {prec:.4f}")
print(f"3. Recall    (재현율) : {rec:.4f}")
print(f"4. F1-Score  (F1점수) : {f1:.4f}\n")

# 6.  리포트 및 혼동 행렬 출력
print("=== 평가 출력 ===")
print(classification_report(y_test, y_pred))

labels = le.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])

print("=== 혼동 행렬 ===")
print(cm_df)