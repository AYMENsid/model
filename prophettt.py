import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === 2. Lissage facultatif ===
df['close'] = df['close'].rolling(window=3).mean()

# === 3. Création des Lags et Indicateurs Techniques ===
df['close_t-1'] = df['close'].shift(1)
df['close_t-2'] = df['close'].shift(2)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
df = df.dropna()

# === 4. Format pour Prophet ===
df_prophet = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})

# === 5. Split train/test (80/20 chronologique) ===
split_idx = int(len(df_prophet) * 0.8)
train = df_prophet[:split_idx]
test = df_prophet[split_idx:]

# === 6. Modèle Prophet avec ajustements ===
# Ajouter des jours fériés
holidays = pd.DataFrame({
  'holiday': 'holiday_name',
  'ds': pd.to_datetime(['2024-12-25', '2024-01-01']),
  'lower_window': 0,
  'upper_window': 1,
})

model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
model.fit(train)

# === 7. Préparer les futures dates ===
future = model.make_future_dataframe(periods=len(test), freq='D')
forecast = model.predict(future)

# === 8. Évaluation sur la portion test ===
y_true = test['y'].values
y_pred = forecast.iloc[split_idx:]['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n Évaluation Prophet :")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# === 9. Visualisation ===
plt.figure(figsize=(12, 6))
plt.plot(test['ds'], y_true, label='Prix réels', color='blue')
plt.plot(test['ds'], y_pred, label='Prix prédits (Prophet)', color='orange')
plt.title(" Prédiction des prix de clôture — Prophet")
plt.xlabel("Date")
plt.ylabel("Prix ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import json

# Sauvegarde des prédictions (CSV)
df_results = pd.DataFrame({
    'date': test['ds'].values,
    'y_true': y_true,
    'y_pred': y_pred
})
df_results.to_csv('prophet_results.csv', index=False)

# Sauvegarde des métriques (JSON)
metrics_prophet = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
with open('prophet_metrics.json', 'w') as f:
    json.dump(metrics_prophet, f)
