import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json
import os

# === 1. Charger et trier les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity', 'date'])

# === 2. Créer le dossier de sortie ===
os.makedirs("xgb_outputs", exist_ok=True)

all_results = []
all_future_results = []  # Pour stocker toutes les prédictions futures
metrics_per_com = {}
maes_train, rmses_train, r2s_train = [], [], []
maes_test, rmses_test, r2s_test = [], [], []

# === 3. Itération par commodity ===
for commodity in df['commodity'].unique():
    sub = df[df['commodity'] == commodity].copy()

    # Indicateurs techniques
    sub['close_smoothed'] = sub['close'].rolling(window=3, min_periods=1).mean()
    sub['close_t-1'] = sub['close_smoothed'].shift(1)
    sub['close_t-2'] = sub['close_smoothed'].shift(2)
    sub['sma_5']      = sub['close_smoothed'].rolling(window=5, min_periods=1).mean()
    sub['ema_10']     = sub['close_smoothed'].ewm(span=10, adjust=False).mean()
    sub['month']      = sub['date'].dt.month
    sub['dayofweek']  = sub['date'].dt.dayofweek
    sub = sub.dropna()

    if len(sub) < 100:
        continue

    # Features et target
    features = ['open','high','low','volume','close_t-1','close_t-2','sma_5','ema_10','month','dayofweek']
    X = sub[features]
    y = sub['close_smoothed']

    # Split chronologique
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = sub['date'].iloc[split_idx:].values

    # Entraînement XGBoost
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Prédiction sur train et test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Évaluation par commodity (train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    maes_train.append(mae_train)
    rmses_train.append(rmse_train)
    r2s_train.append(r2_train)

    # Évaluation par commodity (test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    metrics_per_com[commodity] = {
        'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
        'test': {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}
    }

    maes_test.append(mae_test)
    rmses_test.append(rmse_test)
    r2s_test.append(r2_test)

    print(f"{commodity} -> Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"{commodity} -> Test  MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

    # Stockage des résultats sur test
    df_out = pd.DataFrame({
        'date':      dates_test,
        'commodity': commodity,
        'y_true':    y_test.values,
        'y_pred':    y_test_pred
    })
    all_results.append(df_out)

    # === Prédiction future sur 1 an (365 jours) ===
    n_future_days = 365
    future_preds = []
    last_row = X.iloc[-1:].copy()  # dernière ligne connue

    for i in range(n_future_days):
        # Prédire
        future_pred = model.predict(last_row)[0]
        future_preds.append(future_pred)

        # Mise à jour pour la prochaine prédiction
        new_row = last_row.copy()
        new_row['close_t-2'] = new_row['close_t-1']
        new_row['close_t-1'] = future_pred
        new_row['sma_5'] = (new_row['sma_5'] * 4 + future_pred) / 5  # mise à jour approximative
        new_row['ema_10'] = new_row['ema_10'] * (1-2/(10+1)) + future_pred*(2/(10+1))  # EMA
        new_row['month'] = (last_row['month'] + (last_row['dayofweek'] + 1) // 30) % 12 + 1
        new_row['dayofweek'] = (last_row['dayofweek'] + 1) % 7

        last_row = new_row

    # Génération des dates futures
    future_dates = pd.date_range(start=sub['date'].max() + pd.Timedelta(days=1), periods=n_future_days, freq='D')

    df_future = pd.DataFrame({
        'date': future_dates,
        'commodity': commodity,
        'y_pred_future': future_preds
    })

    all_future_results.append(df_future)
    df_future.to_csv(f"xgb_outputs/{commodity}_future_predictions.csv", index=False)

# === 4. Concaténation et sauvegarde des résultats ===
df_results = pd.concat(all_results, ignore_index=True)
df_results.to_csv('xgb_results.csv', index=False)

# === 5. Concaténation et sauvegarde des prédictions futures ===
df_all_future = pd.concat(all_future_results, ignore_index=True)
df_all_future.to_csv('xgb_outputs/all_future_predictions.csv', index=False)

# === 6. Calcul des métriques globales (moyenne par commodity) ===
global_metrics = {
    'train': {
        'MAE':  float(np.mean(maes_train)),
        'RMSE': float(np.mean(rmses_train)),
        'R2':   float(np.mean(r2s_train))
    },
    'test': {
        'MAE':  float(np.mean(maes_test)),
        'RMSE': float(np.mean(rmses_test)),
        'R2':   float(np.mean(r2s_test))
    }
}

# === 7. Sauvegarde des métriques détaillées et globales ===
with open('xgb_metrics.json', 'w') as f:
    json.dump({
        'by_commodity': metrics_per_com,
        'global': global_metrics
    }, f, indent=2)

# === 8. Sauvegarde des seules métriques globales ===
with open('xgb_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

# === 9. Affichage des métriques globales ===
print("\nMétriques globales (moyenne par commodity) :")
print(f"Train MAE : {global_metrics['train']['MAE']:.4f} | Train RMSE : {global_metrics['train']['RMSE']:.4f} | Train R2 : {global_metrics['train']['R2']:.4f}")
print(f"Test MAE  : {global_metrics['test']['MAE']:.4f} | Test RMSE  : {global_metrics['test']['RMSE']:.4f} | Test R2  : {global_metrics['test']['R2']:.4f}")
