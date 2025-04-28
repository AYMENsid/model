import pandas as pd
import numpy as np
import json
import math
import itertools
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === 1. Charger et trier les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Lissage optionnel
df['y'] = df['close'].rolling(window=3, min_periods=1).mean()
# Régressors techniques
df['lag1'] = df['y'].shift(1)
df['lag2'] = df['y'].shift(2)
df['sma_5'] = df['y'].rolling(5, min_periods=1).mean()
df['ema_10'] = df['y'].ewm(span=10, adjust=False).mean()
df.dropna(inplace=True)

# Prépare DataFrame pour Prophet
df_prophet = pd.DataFrame({
    'ds': df['date'],
    'y': df['y'],
    'lag1': df['lag1'],
    'lag2': df['lag2'],
    'sma_5': df['sma_5'],
    'ema_10': df['ema_10'],
    'commodity': df['commodity']
})

# === 2. Split chronologique 80/20 ===
split = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:split].reset_index(drop=True)
test_df  = df_prophet.iloc[split:].reset_index(drop=True)

# === 3. Hyperparameter grid ===
param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [1.0, 5.0, 10.0, 20.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

best_rmse = float('inf')
best_params = None

# === 4. Grid search ===
for cps, sps, mode in itertools.product(
        param_grid['changepoint_prior_scale'],
        param_grid['seasonality_prior_scale'],
        param_grid['seasonality_mode']
    ):
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        seasonality_mode=mode
    )
    # Ajouter nos régressors
    for reg in ['lag1','lag2','sma_5','ema_10']:
        m.add_regressor(reg)
    
    m.fit(train_df[['ds','y','lag1','lag2','sma_5','ema_10']])
    
    # Prédiction test
    pred = m.predict(test_df[['ds','lag1','lag2','sma_5','ema_10']])
    y_pred = pred['yhat'].values
    y_true = test_df['y'].values
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = {'cps': cps, 'sps': sps, 'mode': mode}

print("Meilleurs hyperparamètres :", best_params, " -- RMSE:", best_rmse)

# === 5. Entraînement final avec les meilleurs paramètres ===
final_model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=best_params['cps'],
    seasonality_prior_scale=best_params['sps'],
    seasonality_mode=best_params['mode']
)
for reg in ['lag1','lag2','sma_5','ema_10']:
    final_model.add_regressor(reg)

final_model.fit(df_prophet)

# === 6. Évaluation finale sur test set ===
pred_test = final_model.predict(test_df[['ds','lag1','lag2','sma_5','ema_10']])
y_pred_test = pred_test['yhat'].values
y_true_test = test_df['y'].values

mae  = mean_absolute_error(y_true_test, y_pred_test)
rmse = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
r2   = r2_score(y_true_test, y_pred_test)

print(f"Final test metrics --- MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# === 7. Sauvegarde des résultats ===
# Fichier 1 : Sauvegarde des prédictions sur l'ensemble de test pour Dash
pd.DataFrame({
    'date': test_df['ds'],
    'y_true': y_true_test,
    'y_pred': y_pred_test
}).to_csv('prophet_results.csv', index=False)

# Fichier 2 : Sauvegarde des métriques globales dans un fichier JSON
global_metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2
}

with open('prophet_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

# === 8. Sauvegarde des métriques par commodity ===
commodities_metrics = {}

for commodity in df['commodity'].unique():
    commodity_data = df[df['commodity'] == commodity]
    true_values = commodity_data['close'].values
    pred_values = final_model.predict(commodity_data[['ds','lag1','lag2','sma_5','ema_10']])['yhat'].values
    
    mae_com = mean_absolute_error(true_values, pred_values)
    rmse_com = math.sqrt(mean_squared_error(true_values, pred_values))
    r2_com = r2_score(true_values, pred_values)
    
    commodities_metrics[commodity] = {
        'MAE': mae_com,
        'RMSE': rmse_com,
        'R2': r2_com
    }

with open('prophet_metrics.json', 'w') as f:
    json.dump(commodities_metrics, f, indent=2)

# === 9. Prédiction future ===
future = final_model.make_future_dataframe(periods=365, freq='D')
# Construire les caractéristiques (lag, sma, ema) pour les jours futurs
future_feat = future.copy()
future_feat['lag1'] = future_feat['y'].shift(1)
future_feat['lag2'] = future_feat['y'].shift(2)
future_feat['sma_5'] = future_feat['y'].rolling(5, min_periods=1).mean()
future_feat['ema_10'] = future_feat['y'].ewm(span=10, adjust=False).mean()

# On prédit avec le modèle final
future_pred = final_model.predict(future_feat[['ds', 'lag1', 'lag2', 'sma_5', 'ema_10']])

# Fichier 4 : Sauvegarde des prédictions futures dans un fichier CSV
pd.DataFrame({
    'date': future_pred['ds'],
    'y_pred': future_pred['yhat']
}).to_csv('prophet_future_result.csv', index=False)
