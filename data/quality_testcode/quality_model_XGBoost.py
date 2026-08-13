import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from xgboost import XGBRegressor
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

# 3. XGBoost 회귀 모델 생성 및 학습 
model_xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model_xgb.fit(X_train, y_train)

print('=== XGBoost Regressor 학습 완료 ===\n')

# 4. 예측 수행
y_pred = model_xgb.predict(X_test)

# 5. 회귀 성능 지표 평가 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== XGBoost Regressor 핵심 성능 지표 ===")
print(f"1. MAE  (평균 절대 오차) : {mae:.4f}")
print(f"2. MSE  (평균 제곱 오차) : {mse:.4f}")
print(f"3. RMSE (제곱근 평균 오차): {rmse:.4f}")
print(f"4. R² Score (결정계수)   : {r2:.4f}\n")

# 6. 실제값 vs 예측값 오차 샘플 확인
y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
pred_result_df = pd.DataFrame({
    'Actual_Quality': y_test_values,
    'Predicted_Quality': y_pred,
    'Residual': y_test_values - y_pred
})

print("=== 실제값 vs 예측값 샘플 (Top 5) ===")
print(pred_result_df.head())
print("\n")

# 7. 서비스용 model/ 폴더 지정 및 최종 모델 덮어쓰기 저장
save_dir = 'model'
os.makedirs(save_dir, exist_ok=True)

final_data = {
    'model': model_xgb,
    'preprocessor': preprocessor
}

# model/quality_model_xgb.pkl (또는 사용하시려는 파일명)으로 저장
save_path = os.path.join(save_dir, 'quality_model_xgb.pkl')
joblib.dump(final_data, save_path)

print(f"=== model 폴더 내 최신 XGBoost 모델 덮어쓰기 완료: {save_path} ===")