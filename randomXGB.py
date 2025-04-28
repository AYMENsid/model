import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import math

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === 2. Feature Engineering ===
df['close'] = df['close'].rolling(window=3).mean()
df['close_t-1'] = df['close'].shift(1)
df['close_t-2'] = df['close'].shift(2)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['ema_10'] = df['close'].ewm(span=10).mean()
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df = df.dropna()

# === 3. Features & Target ===
features = ['open', 'high', 'low', 'volume', 'close_t-1', 'close_t-2', 'sma_5', 'ema_10', 'month', 'dayofweek']
X = df[features]
y = df['close']

# === 4. Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)

# === 5. Stacking Model ===
base_models = [
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6))
]

stack_model = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0))
stack_model.fit(X_train, y_train)

# === 6. Prédiction ===
y_pred = stack_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nStacking Model (RF + XGBoost)")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# === 7. Visualisation ===
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Prix réels')
plt.plot(y_pred, label='Prix prédits (Stacking)', color='purple')
plt.title('Prédiction des prix - Stacking RandomForest + XGBoost')
plt.xlabel('Jours')
plt.ylabel('Prix ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. Export CSV/JSON ===
test_dates = df['date'].values[-len(y_test):]
df_stack = pd.DataFrame({
    'date': test_dates,
    'y_true': y_test.values,
    'y_pred': y_pred
})
df_stack.to_csv('stacking_results.csv', index=False)

with open('stacking_metrics.json', 'w') as f:
    json.dump({'MAE': mae, 'RMSE': rmse, 'R2': r2}, f)
