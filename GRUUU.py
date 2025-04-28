from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import json
import os
import math

# === 1. Charger et trier les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity', 'date'])

# === 2. Créer dossier de sortie ===
os.makedirs("gru_outputs", exist_ok=True)

all_results = []
metrics_per_com = {}

# === 3. Itération par commodity ===
for commodity in df['commodity'].unique():
    sub = df[df['commodity'] == commodity].copy()
    sub = sub.dropna(subset=['close'])
    if len(sub) < 100:
        continue

    # === Normalisation ===
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sub['close'].values.reshape(-1,1))

    # === Séquences ===
    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i,0])
        y.append(scaled[i,0])
    X, y = np.array(X), np.array(y)
    if len(X) < 100:
        continue
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # === Split ===
    split_idx = int(len(X)*0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = sub['date'].values[seq_len:][split_idx:]

    # === Modèle GRU ===
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(seq_len,1)),
        Dropout(0.2),
        GRU(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # === Prédictions ===
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_inv = scaler.inverse_transform(y_train_pred)
    y_test_inv = scaler.inverse_transform(y_test_pred)

    # === Metrics par commodity ===
    mae_train = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1,1)), y_train_inv)
    rmse_train = math.sqrt(mean_squared_error(scaler.inverse_transform(y_train.reshape(-1,1)), y_train_inv))
    r2_train = r2_score(scaler.inverse_transform(y_train.reshape(-1,1)), y_train_inv)

    mae_test = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1,1)), y_test_inv)
    rmse_test = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), y_test_inv))
    r2_test = r2_score(scaler.inverse_transform(y_test.reshape(-1,1)), y_test_inv)

    metrics_per_com[commodity] = {
        'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
        'test': {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}
    }

    print(f"{commodity} -> Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"{commodity} -> Test  MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

    # === Stockage des prédictions détaillées ===
    df_out = pd.DataFrame({
        'date': dates_test,
        'commodity': commodity,
        'y_true': scaler.inverse_transform(y_test.reshape(-1,1)).flatten(),
        'y_pred': y_test_inv.flatten()
    })
    all_results.append(df_out)

# === 4. Concaténation et sauvegarde du CSV détaillé ===
df_results = pd.concat(all_results, ignore_index=True)
df_results.to_csv('gru_results.csv', index=False)

# === 5. Sauvegarde des métriques par commodity en CSV ===
# Aplatir les données des métriques
flattened_metrics = []
for commodity, metrics in metrics_per_com.items():
    flattened_metrics.append({
        'commodity': commodity,
        'train_MAE': metrics['train']['MAE'],
        'train_RMSE': metrics['train']['RMSE'],
        'train_R2': metrics['train']['R2'],
        'test_MAE': metrics['test']['MAE'],
        'test_RMSE': metrics['test']['RMSE'],
        'test_R2': metrics['test']['R2']
    })

# Créer le DataFrame à partir des données aplaties
metrics_df = pd.DataFrame(flattened_metrics)

# Sauvegarder les métriques
metrics_df.to_csv('gru_metrics.csv', index=False)

# === 6. Calcul et sauvegarde des métriques globales ===
global_metrics = {
    'train': {
        'MAE':  float(np.mean([metrics['train']['MAE'] for metrics in metrics_per_com.values()])),
        'RMSE': float(np.mean([metrics['train']['RMSE'] for metrics in metrics_per_com.values()])),
        'R2':   float(np.mean([metrics['train']['R2'] for metrics in metrics_per_com.values()]))
    },
    'test': {
        'MAE':  float(np.mean([metrics['test']['MAE'] for metrics in metrics_per_com.values()])),
        'RMSE': float(np.mean([metrics['test']['RMSE'] for metrics in metrics_per_com.values()])),
        'R2':   float(np.mean([metrics['test']['R2'] for metrics in metrics_per_com.values()]))
    }
}

with open('gru_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f)

print("\nMétriques globales :")
print(f"Train MAE : {global_metrics['train']['MAE']:.4f} | Train RMSE : {global_metrics['train']['RMSE']:.4f} | Train R2 : {global_metrics['train']['R2']:.4f}")
print(f"Test MAE  : {global_metrics['test']['MAE']:.4f} | Test RMSE  : {global_metrics['test']['RMSE']:.4f} | Test R2  : {global_metrics['test']['R2']:.4f}")
