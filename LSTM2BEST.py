import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
import json

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === 2. Prétraitement ===
features = ['open', 'high', 'low', 'close', 'volume']
data = df[features].values

# Normaliser toutes les colonnes
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Préparation des séquences
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i])          # (60, 5)
    y.append(scaled_data[i, features.index('close')])     # seul le close comme cible

X, y = np.array(X), np.array(y)

# Split train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === 3. Modèle LSTM Multivarié ===
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

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# === 4. Évaluation ===
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# Inverse de la normalisation
scaler_close = MinMaxScaler()
close_only = df[['close']].values
scaler_close.fit(close_only)

predicted_train_prices = scaler_close.inverse_transform(predicted_train)
predicted_test_prices = scaler_close.inverse_transform(predicted_test)
real_train_prices = scaler_close.inverse_transform(y_train.reshape(-1, 1))
real_test_prices = scaler_close.inverse_transform(y_test.reshape(-1, 1))

# Calcul des métriques
rmse_train = math.sqrt(mean_squared_error(real_train_prices, predicted_train_prices))
mae_train = mean_absolute_error(real_train_prices, predicted_train_prices)
r2_train = r2_score(real_train_prices, predicted_train_prices)

rmse_test = math.sqrt(mean_squared_error(real_test_prices, predicted_test_prices))
mae_test = mean_absolute_error(real_test_prices, predicted_test_prices)
r2_test = r2_score(real_test_prices, predicted_test_prices)

# Affichage des métriques pour le train et le test
print(f"\n--- Entraînement ---")
print(f"RMSE: {rmse_train:.4f}")
print(f"MAE : {mae_train:.4f}")
print(f"R²  : {r2_train:.4f}")

print(f"\n--- Test ---")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE : {mae_test:.4f}")
print(f"R²  : {r2_test:.4f}")

# === 5. Visualisation des prédictions ===
plt.figure(figsize=(12, 6))
plt.plot(real_test_prices, label='Prix réels (Test)')
plt.plot(predicted_test_prices, label='Prix prédits (Test)')
plt.title('Prédiction du prix de clôture du pétrole (Crude Oil)')
plt.xlabel('Jours')
plt.ylabel('Prix ($)')
plt.legend()
plt.grid(True)
plt.show()

# === 6. Courbe de perte ===
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title("Courbe de perte (loss)")
plt.xlabel("Épochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# === 7. Sauvegarde des résultats ===
# Créer un vecteur de dates correspondant à la taille de y_test
test_dates = df['date'].values[-len(y_test):]

# Construire le DataFrame de résultats
df_lstm = pd.DataFrame({
    'date': test_dates,
    'y_true': real_test_prices.flatten(),
    'y_pred': predicted_test_prices.flatten()
})
df_lstm.to_csv('lstm_results.csv', index=False)

# === 8. Sauvegarde des métriques ===
metrics_lstm = {
    'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
    'test': {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}
}
with open('lstm_metrics.json', 'w') as f:
    json.dump(metrics_lstm, f, indent=2)

# === 9. Sauvegarde des seules métriques globales ===
global_metrics = {
    'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
    'test': {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}
}
with open('lstm_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

# === 10. Affichage des métriques globales ===
print("\nMétriques globales (moyenne par dataset) :")
print(f"Train MAE : {metrics_lstm['train']['MAE']:.4f} | Train RMSE : {metrics_lstm['train']['RMSE']:.4f} | Train R2 : {metrics_lstm['train']['R2']:.4f}")
print(f"Test  MAE : {metrics_lstm['test']['MAE']:.4f} | Test  RMSE : {metrics_lstm['test']['RMSE']:.4f} | Test  R2 : {metrics_lstm['test']['R2']:.4f}")
