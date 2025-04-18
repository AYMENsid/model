import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math

# === Charger et préparer les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Moyenne mobile pour lisser la série
df['close'] = df['close'].rolling(window=3).mean()
df = df.dropna()

# === Normalisation ===
features = ['open', 'high', 'low', 'close', 'volume']
data = df[features].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# === Séquences ===
def create_sequences(data, target_idx, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, features.index('close'))

# Train/test split
split = int(len(X) * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Fonction pour créer un modèle avec activation donnée ===
def build_model(activation):
    model = Sequential([
        LSTM(128, activation=activation, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        LSTM(64, activation=activation, return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation=activation),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

# === Entraînement et évaluation de chaque modèle ===
activations = ['relu', 'tanh', 'sigmoid']
results = {}

for act in activations:
    print(f"\n=== Entraînement avec activation : {act.upper()} ===")
    model = build_model(act)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    pred = model.predict(X_test)
    
    # Inverse transformation uniquement sur 'close'
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[['close']].values)
    pred_prices = scaler_close.inverse_transform(pred)
    real_prices = scaler_close.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = math.sqrt(mean_squared_error(real_prices, pred_prices))
    mae = mean_absolute_error(real_prices, pred_prices)
    r2 = r2_score(real_prices, pred_prices)

    results[act] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

    # Courbe des prix (facultative)
    plt.figure(figsize=(10,4))
    plt.plot(real_prices, label='Réel')
    plt.plot(pred_prices, label='Prédit')
    plt.title(f"Activation : {act.upper()} — MAE={mae:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

# === Affichage des résultats ===
print("\n=== Résultats Comparatifs ===")
print(f"{'Activation':<10} | {'MAE':<8} | {'RMSE':<8} | {'R²':<8}")
print("-" * 40)
for act, metrics in results.items():
    print(f"{act:<10} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f} | {metrics['R2']:.4f}")
