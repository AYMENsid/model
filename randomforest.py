import pandas as pd
import numpy as np
import math
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity', 'date'])

# === Créer dossier résultats ===
os.makedirs("rf_outputs", exist_ok=True)

all_results = []
metrics_per_commodity = {}
maes_train, rmses_train, r2s_train = [], [], []
maes_test, rmses_test, r2s_test = [], [], []

# === Parcours des commodités ===
for commodity in df['commodity'].unique():
    sub = df[df['commodity'] == commodity].copy()

    # Feature engineering
    sub['close_smoothed'] = sub['close'].rolling(window=3, min_periods=1).mean()
    sub['close_t-1'] = sub['close_smoothed'].shift(1)
    sub['close_t-2'] = sub['close_smoothed'].shift(2)
    sub['sma_5'] = sub['close_smoothed'].rolling(window=5, min_periods=1).mean()
    sub['ema_10'] = sub['close_smoothed'].ewm(span=10, adjust=False).mean()
    sub['month'] = sub['date'].dt.month
    sub['dayofweek'] = sub['date'].dt.dayofweek
    sub = sub.dropna()

    if len(sub) < 100:
        continue

    features = ['open', 'high', 'low', 'volume', 'close_t-1', 'close_t-2', 'sma_5', 'ema_10', 'month', 'dayofweek']
    target = 'close_smoothed'

    X = sub[features]
    y = sub[target]
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = sub['date'].iloc[split_idx:].values

    # Entraînement RF
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Prédiction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Évaluation (train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train = r2_score(y_train, y_train_pred)
    maes_train.append(mae_train)
    rmses_train.append(rmse_train)
    r2s_train.append(r2_train)

    # Évaluation (test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)
    maes_test.append(mae_test)
    rmses_test.append(rmse_test)
    r2s_test.append(r2_test)

    metrics_per_commodity[commodity] = {
        'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
        'test': {'MAE': mae_test, 'RMSE': mae_test, 'R2': r2_test}
    }

    print(f"{commodity} -> Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"{commodity} -> Test  MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

    # Stockage des résultats test
    df_out = pd.DataFrame({
        'date': dates_test,
        'commodity': commodity,
        'y_true': y_test.values,
        'y_pred': y_test_pred
    })
    all_results.append(df_out)

# === Sauvegarde des résultats détaillés ===
df_results = pd.concat(all_results, ignore_index=True)
df_results.to_csv('rf_results.csv', index=False)

# === Calcul des métriques globales ===
global_metrics = {
    'train': {
        'MAE': float(np.mean(maes_train)),
        'RMSE': float(np.mean(rmses_train)),
        'R2': float(np.mean(r2s_train))
    },
    'test': {
        'MAE': float(np.mean(maes_test)),
        'RMSE': float(np.mean(rmses_test)),
        'R2': float(np.mean(r2s_test))
    }
}

# === Sauvegarde métriques par commodity ===
with open('rf_metrics.json', 'w') as f:
    json.dump({
        'by_commodity': metrics_per_commodity,
        'global': global_metrics
    }, f, indent=2)

# === Sauvegarde métriques globales seules ===
with open('rf_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

# === Affichage résumé ===
print("\nMÉTRIQUES GLOBALES :")
print(f"Train MAE : {global_metrics['train']['MAE']:.4f} | Train RMSE : {global_metrics['train']['RMSE']:.4f} | Train R2 : {global_metrics['train']['R2']:.4f}")
print(f"Test MAE  : {global_metrics['test']['MAE']:.4f} | Test RMSE  : {global_metrics['test']['RMSE']:.4f} | Test R2  : {global_metrics['test']['R2']:.4f}")
