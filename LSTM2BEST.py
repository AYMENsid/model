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
predicted = model.predict(X_test)

# Récupérer uniquement l’échelle du 'close' pour inverse_transform
scaler_close = MinMaxScaler()
close_only = df[['close']].values
scaler_close.fit(close_only)

predicted_prices = scaler_close.inverse_transform(predicted)
real_prices = scaler_close.inverse_transform(y_test.reshape(-1, 1))

# Erreurs
rmse = math.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)

print(f"\n RMSE: {rmse:.4f}")
print(f" MAE : {mae:.4f}")
print(f"R²: {r2:.4f}")

# === 5. Visualisation des prédictions ===
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Prix réels')
plt.plot(predicted_prices, label='Prix prédits')
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

import json

# Créer un vecteur de dates correspondant à la taille de y_test
# On prend les dates à partir de l'index du DataFrame original
test_dates = df['date'].values[-len(y_test):]

# Construire le DataFrame de résultats
df_lstm = pd.DataFrame({
    'date': test_dates,
    'y_true': real_prices.flatten(),
    'y_pred': predicted_prices.flatten()
})
df_lstm.to_csv('lstm_results.csv', index=False)

# Calcul du R²
from sklearn.metrics import r2_score
r2 = r2_score(real_prices, predicted_prices)

# Sauvegarde des métriques
metrics_lstm = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
with open('lstm_metrics.json', 'w') as f:
    json.dump(metrics_lstm, f)
