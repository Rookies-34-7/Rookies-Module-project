import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# sklearn 텍스트 출력 설정
sklearn.set_config(display='text')

# 1. 전처리된 데이터 로드
df = joblib.load('quality_prep(1).pkl')

# 2. 데이터 추출
X_train = df['X_train']
X_test = df['X_test']
y_train_raw = df['y_train']  # 문자열 데이터 ('Low', 'Medium', 'High')
y_test_raw = df['y_test']
preprocessor = df.get('preprocessor', None)

# 문자열 라벨('Low', 'Medium', 'High')을 수치(0, 1, 2)로 변환

label_map = {'Low': 0, 'Medium': 1, 'High': 2}

# 문자열을 숫자로 매핑 
if isinstance(y_train_raw.iloc[0] if hasattr(y_train_raw, 'iloc') else y_train_raw[0], str):
    y_train = y_train_raw.map(label_map) if hasattr(y_train_raw, 'map') else pd.Series(y_train_raw).map(label_map)
    y_test = y_test_raw.map(label_map) if hasattr(y_test_raw, 'map') else pd.Series(y_test_raw).map(label_map)
else:
    y_train = y_train_raw
    y_test = y_test_raw

# 3. 선형 회귀(Linear Regression) 모델 생성 및 학습
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)  # 이제 숫자로 변환되었으므로 정상 학습됨!

print("=== 학습 완료 ===\n")

# 4. 예측 수행
y_pred = model.predict(X_test)

# 5. 회귀 성능 평가 지표 계산
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== 선형 회귀 핵심 성능 지표 ===")
print(f"1. MAE  (평균 절대 오차) : {mae:.4f}")
print(f"2. MSE  (평균 제곱 오차) : {mse:.4f}")
print(f"3. RMSE (제곱근 평균 오차): {rmse:.4f}")
print(f"4. R² Score (결정계수)   : {r2:.4f}\n")

# 6. 피처별 회귀 계수(Feature Coefficients) 추출
if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
    feature_names = preprocessor.get_feature_names_out()
else:
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("=== 피처별 회귀 계수 (Feature Coefficients) ===")
print(coef_df)
print("\n")
