import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix, precision_score


import sklearn
sklearn.set_config(display='text')


# 1. 데이터 로드
df = joblib.load('../quality_data/quality_prep.pkl')


# 2. 저장된 변수들 추출
X_train = df['X_train']
X_test = df['X_test']
y_train = df['y_train']
y_test = df['y_test']
preprocessor = df['preprocessor']  


# 3. 랜덤 포레스트 분류기 생성 및 학습
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',   
    n_jobs=-1
)

model.fit(X_train, y_train)

print('===학습 완료===')


# 4. 예측 및 평가
y_pred = model.predict(X_test)

# 다중 클래스(High, Medium, Low) 평가이므로 average='weighted' 지정
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("=== 모델 정확도 ===")
print(f" Accuracy  (정확도) : {acc:.4f}")

print("=== 핵심 성능 지표 ===")
print(f"1. Accuracy  (정확도) : {acc:.4f}")
print(f"2. Precision (정밀도) : {prec:.4f}")
print(f"3. Recall    (재현율) : {rec:.4f}")
print(f"4. F1-Score  (F1점수) : {f1:.4f}\n")


# 평가 출력
print("=== 분류 결과 상세 리포트 ===")
print(classification_report(y_test, y_pred))


# Confusion Matrix 텍스트
# label 순서 추출
labels = model.classes_
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("=== 혼동 행렬 (Confusion Matrix) ===")
cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
print(cm_df)
print("\n")


# CSV 저장
cm_df.to_csv('../quality_data/quality_confusionmatrix.csv', index=True, encoding='utf-8-sig')


# pkl 저장
joblib.dump(model, '../quality_data/quality_model_info.pkl')


final_data = {
    'model': model,
    'preprocessor': preprocessor,
    'labels': labels
}
joblib.dump(final_data, '../quality_data/quality_model_info.pkl')


