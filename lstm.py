import pandas as pd
import numpy as np
import os
import json
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity', 'date'])

# === 2. Créer dossier de sortie ===
os.makedirs("lstm_outputs", exist_ok=True)

all_results = []
all_future_preds = []  # <== pour stocker les prédictions futures
metrics_per_com = {}

# === 3. Itération par commodity ===
for commodity in df['commodity'].unique():
    sub = df[df['commodity'] == commodity].copy()
    sub = sub.dropna(subset=['close'])
    if len(sub) < 100:
        continue

    # Normaliser toutes les colonnes
    features = ['open', 'high', 'low', 'close', 'volume']
    data = sub[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Séquences
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, features.index('close')])
    X, y = np.array(X), np.array(y)
    if len(X) < 100:
        continue

    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = sub['date'].values[sequence_length:][split_idx:]

    # === Modèle LSTM ===
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60, 5)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

    # Prédictions sur test
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Inverse transform close
    scaler_close = MinMaxScaler()
    scaler_close.fit(sub[['close']].values)
    y_pred_test_inv = scaler_close.inverse_transform(y_pred_test)
    y_pred_train_inv = scaler_close.inverse_transform(y_pred_train)
    y_test_inv = scaler_close.inverse_transform(y_test.reshape(-1, 1))
    y_train_inv = scaler_close.inverse_transform(y_train.reshape(-1, 1))

    # Métriques
    mae_test = mean_absolute_error(y_test_inv, y_pred_test_inv)
    rmse_test = math.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    r2_test = r2_score(y_test_inv, y_pred_test_inv)

    mae_train = mean_absolute_error(y_train_inv, y_pred_train_inv)
    rmse_train = math.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    r2_train = r2_score(y_train_inv, y_pred_train_inv)

    print(f"{commodity} -> Test: MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")
    print(f"{commodity} -> Train: MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")

    metrics_per_com[commodity] = {
        'MAE_train': mae_train,
        'RMSE_train': rmse_train,
        'R2_train': r2_train,
        'MAE_test': mae_test,
        'RMSE_test': rmse_test,
        'R2_test': r2_test
    }

    # Stockage prédictions test
    df_out = pd.DataFrame({
        'date': dates_test,
        'commodity': commodity,
        'y_true': y_test_inv.flatten(),
        'y_pred': y_pred_test_inv.flatten()
    })
    all_results.append(df_out)

    # === Prédiction FUTURE pour 1 an ===
    n_future_days = 365
    last_60_days = scaled_data[-60:]  # Les dernières séquences connues
    future_preds = []

    for _ in range(n_future_days):
        X_pred = np.expand_dims(last_60_days, axis=0)  # (1, 60, 5)
        future_pred_scaled = model.predict(X_pred, verbose=0)
        future_pred_close = scaler_close.inverse_transform(future_pred_scaled)[0][0]
        future_preds.append(future_pred_close)

        # Créer nouvelle entrée pour next time step
        new_entry = np.copy(last_60_days[-1])
        new_entry[features.index('close')] = future_pred_scaled  # mettre la close prédite
        last_60_days = np.vstack((last_60_days[1:], new_entry))  # slide window

    # Générer dates futures
    last_date = sub['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future_days)

    # Stockage des résultats futures
    df_future = pd.DataFrame({
        'date': future_dates,
        'commodity': commodity,
        'predicted_close': future_preds
    })
    all_future_preds.append(df_future)

# === 4. Sauvegarde des résultats ===

# Prédictions sur données test
df_results = pd.concat(all_results, ignore_index=True)
df_results.to_csv('lstm_results.csv', index=False)

# Prédictions futures
df_future_results = pd.concat(all_future_preds, ignore_index=True)
df_future_results.to_csv('predict_lstm.csv', index=False)

# Métriques par commodity
with open('lstm_metrics_with_train_test.json', 'w') as f:
    json.dump(metrics_per_com, f, indent=4)

# Métriques globales
mae_test_avg = np.mean([metrics['MAE_test'] for metrics in metrics_per_com.values()])
rmse_test_avg = np.mean([metrics['RMSE_test'] for metrics in metrics_per_com.values()])
r2_test_avg = np.mean([metrics['R2_test'] for metrics in metrics_per_com.values()])
mae_train_avg = np.mean([metrics['MAE_train'] for metrics in metrics_per_com.values()])
rmse_train_avg = np.mean([metrics['RMSE_train'] for metrics in metrics_per_com.values()])
r2_train_avg = np.mean([metrics['R2_train'] for metrics in metrics_per_com.values()])

global_metrics = {
    'MAE_test': mae_test_avg,
    'RMSE_test': rmse_test_avg,
    'R2_test': r2_test_avg,
    'MAE_train': mae_train_avg,
    'RMSE_train': rmse_train_avg,
    'R2_train': r2_train_avg
}

with open('lstm_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f)

print("\nMétriques globales :")
print(f"MAE (test) : {global_metrics['MAE_test']:.4f} | RMSE (test) : {global_metrics['RMSE_test']:.4f} | R2 (test) : {global_metrics['R2_test']:.4f}")
print(f"MAE (train) : {global_metrics['MAE_train']:.4f} | RMSE (train) : {global_metrics['RMSE_train']:.4f} | R2 (train) : {global_metrics['R2_train']:.4f}")
