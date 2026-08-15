import joblib
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sklearn.set_config(display='text')

# 1. 범주화 전 데이터 로드
pkl_path = 'data/quality_testcode/quality_prep_original_target.pkl'
df = joblib.load(pkl_path)

# 2. 데이터 및 전처리기 추출
X_train = df['X_train']
X_test = df['X_test']
y_train = df['y_train']  # 연속형 수치 데이터 (예: 4~10점)
y_test = df['y_test']
preprocessor = df.get('preprocessor', None)

# 3. 랜덤 포레스트 회귀 모델 생성 및 학습
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print('=== 학습 완료 ===\n')

# 4. 예측 수행
y_pred = model.predict(X_test)

# 5. 회귀 성능 지표 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Random Forest Regressor 핵심 성능 지표 ===")
print(f"1. MAE  (평균 절대 오차) : {mae:.4f}")
print(f"2. MSE  (평균 제곱 오차) : {mse:.4f}")
print(f"3. RMSE (제곱근 평균 오차): {rmse:.4f}")
print(f"4. R² Score (결정계수)   : {r2:.4f}\n")

# 6. 평가 지표 및 예측 결과 데이터프레임 생성
# (1) 성능 지표 데이터프레임
metrics_df = pd.DataFrame([{
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2_Score': r2
}])

# (2) 실제값 vs 예측값 및 오차 데이터프레임
y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
pred_result_df = pd.DataFrame({
    'Actual_Quality': y_test_values,
    'Predicted_Quality': y_pred,
    'Residual': y_test_values - y_pred
})

print("=== 실제값 vs 예측값 샘플 (Top 5) ===")
print(pred_result_df.head())
print("\n")

# 파일 저장
test_save_dir = 'data/quality_testcode'
model_save_dir = 'model'

save_data = {
    'model': model,
    'preprocessor': preprocessor
}


final_model_path = os.path.join(model_save_dir, 'quality_model_rf.pkl')
joblib.dump(save_data, final_model_path)

print(f"-> 회귀 평가 지표 및 예측 결과 저장 완료: {test_save_dir}")
print(f"-> 최신 랜덤 포레스트 모델 덮어쓰기 완료: {final_model_path}")
