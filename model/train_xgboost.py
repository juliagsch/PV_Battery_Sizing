from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import xgboost as xgb
import joblib

base_path = "."

# Preparing data
data = np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_train.csv', delimiter=",")
X = data[:, :-2]
y = data[:, -2:]  # optimal battery and solar sizing

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train two separate regressors for battery and solar sizing predictions
battery_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
solar_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)

battery_model.fit(X_train, y_train[:, 0])
solar_model.fit(X_train, y_train[:, 1])

# Validation predictions
battery_pred = battery_model.predict(X_val)
solar_pred = solar_model.predict(X_val)

# Test model
battery_mse = mean_squared_error(y_val[:, 0], battery_pred)
battery_mae = mean_absolute_error(y_val[:, 0], battery_pred)
solar_mse = mean_squared_error(y_val[:, 1], solar_pred)
solar_mae = mean_absolute_error(y_val[:, 1], solar_pred)

print(f"Validation MSE - Battery: {battery_mse:.4f}, Solar: {solar_mse:.4f}")
print(f"Validation MAE - Battery: {battery_mae:.4f}, Solar: {solar_mae:.4f}")

# Save models and scaler
joblib.dump(battery_model, f"{base_path}/battery_model_xgb.pkl")
joblib.dump(solar_model, f"{base_path}/solar_model_xgb.pkl")
joblib.dump(scaler, f"{base_path}/scaler_xgb.pkl")

# Final evaluation on test set
data_test = np.loadtxt(f'{base_path}/dataset/dataset_below_threshold_test.csv', delimiter=",")
X_test = data_test[:, :-2]
y_test = data_test[:, -2:]
X_test = scaler.transform(X_test)

battery_test_pred = battery_model.predict(X_test)
solar_test_pred = solar_model.predict(X_test)

battery_test_mse = mean_squared_error(y_test[:, 0], battery_test_pred)
battery_test_mae = mean_absolute_error(y_test[:, 0], battery_test_pred)
solar_test_mse = mean_squared_error(y_test[:, 1], solar_test_pred)
solar_test_mae = mean_absolute_error(y_test[:, 1], solar_test_pred)

print(f"Test MSE Battery: {battery_test_mse:.4f}, Solar: {solar_test_mse:.4f}")
print(f"Test MAE Battery: {battery_test_mae:.4f}, Solar: {solar_test_mae:.4f}")
