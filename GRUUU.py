from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données (remplace ceci par ton propre dataset)
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Prétraitement des données (utilisation du 'close')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

# Création des séquences pour l'entrée (séquence de 60 jours)
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])  # Séquences de 60 jours
    y.append(scaled_data[i, 0])  # Valeur de la cible (le prix à prédire)
X, y = np.array(X), np.array(y)

# Reshaper pour que le modèle GRU accepte l'entrée
X = X.reshape((X.shape[0], X.shape[1], 1))  # 3D (échantillons, timesteps, features)

# Split train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Construire le modèle GRU
model = Sequential()

model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Prédictions
predicted_prices = model.predict(X_test)

# Inverser la normalisation
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted_prices = scaler.inverse_transform(predicted_prices)

# Évaluation
rmse = np.sqrt(np.mean(np.square(real_prices - predicted_prices)))
mae = np.mean(np.abs(real_prices - predicted_prices))
r2 = r2_score(real_prices, predicted_prices)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label="Prix réels")
plt.plot(predicted_prices, label="Prix prédits (GRU)", color='red')
plt.title("Prédiction des prix de clôture du pétrole avec GRU")
plt.xlabel("Jours")
plt.ylabel("Prix ($)")
plt.legend()
plt.grid(True)
plt.show()
import json

# Récupérer les dates correspondant à y_test
test_dates = df['date'].values[-len(y_test):]

# Créer le DataFrame des résultats GRU
df_gru = pd.DataFrame({
    'date': test_dates,
    'y_true': real_prices.flatten(),
    'y_pred': predicted_prices.flatten()
})
df_gru.to_csv('gru_results.csv', index=False)

# Sauvegarder les métriques
metrics_gru = {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
with open('gru_metrics.json', 'w') as f:
    json.dump(metrics_gru, f)
