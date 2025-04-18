import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

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
close_prices = df[['close']].values
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_close)):
    X.append(scaled_close[i - sequence_length:i])
    y.append(scaled_close[i])
X, y = np.array(X), np.array(y)

# Reshape si nécessaire
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === 3. Modèle LSTM Amélioré ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 1)),
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
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test)

rmse = math.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)  # <-- Calcul du R²
print(f"\n RMSE: {rmse:.4f}")
print(f" MAE : {mae:.4f}")
print(f" R²  : {r2:.4f}")  # <-- Affichage du R²
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

# === 6. Visualisation de la perte ===
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Perte entraînement')
plt.plot(history.history['val_loss'], label='Perte validation')
plt.title("Courbe de perte (loss)")
plt.xlabel("Épochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
