import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Charger les données
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', inplace=True)

# Fonction pour créer les séquences temporelles
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Résultats à sauvegarder
results = []
future_predictions = []
metrics = {}
global_y_true, global_y_pred = [], []

# Paramètres
window_size = 60
future_days = 365

for commodity in df['commodity'].unique():
    print(f"\nTraitement de la commodity : {commodity}")
    df_c = df[df['commodity'] == commodity].copy()
    df_c.set_index('date', inplace=True)
    prices = df_c['close'].values.reshape(-1, 1)

    # Normalisation
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    # Train/test split
    train_size = int(len(prices_scaled) * 0.8)
    train, test = prices_scaled[:train_size], prices_scaled[train_size:]

    # Séquences
    X_train, y_train = create_sequences(train, window_size)
    X_test,  y_test  = create_sequences(test,  window_size)

    # Reshape
    X_train = X_train.reshape((len(X_train), window_size, 1))
    X_test  = X_test.reshape((len(X_test),  window_size, 1))

    # Modèle LSTM avec Dropout
    model = Sequential([
        LSTM(100, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_check = ModelCheckpoint(f"{commodity}_best_model.h5", save_best_only=True, monitor='val_loss')

    # Entraînement
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_split=0.1, callbacks=[early_stop, model_check])

    # Prédictions test
    y_pred_test_s = model.predict(X_test)
    y_test_inv    = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_test   = scaler.inverse_transform(y_pred_test_s).flatten()

    # Prédictions train
    y_pred_train_s = model.predict(X_train)
    y_train_inv    = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
    y_pred_train   = scaler.inverse_transform(y_pred_train_s).flatten()

    # Enregistrer prédictions test (brut)
    dates_test = df_c.index[-len(y_test):]
    for dt, actual, pred in zip(dates_test, y_test_inv, y_pred_test):
        results.append({
            'date': dt.date(), 
            'commodity': commodity, 
            'y_true': actual, 
            'y_pred': pred
        })

    # Pour calcul global
    global_y_true.extend(y_test_inv)
    global_y_pred.extend(y_pred_test)

    # Prédictions futures
    seq = prices_scaled[-window_size:].reshape(1, window_size, 1)
    future_scaled = []
    for _ in range(future_days):
        p = model.predict(seq)[0,0]
        future_scaled.append(p)
        seq = np.append(seq[:,1:,:], [[[p]]], axis=1)

    future_pred = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()
    start_date  = df_c.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=future_days)

    for dt, pred in zip(future_dates, future_pred):
        future_predictions.append({
            'commodity': commodity,
            'date': dt.date(),
            'predicted': pred
        })

    # Métriques
    mae_t  = mean_absolute_error(y_test_inv, y_pred_test)
    rmse_t = np.sqrt(mean_squared_error(y_test_inv, y_pred_test))
    r2_t   = r2_score(y_test_inv, y_pred_test)

    mae_tr  = mean_absolute_error(y_train_inv, y_pred_train)
    rmse_tr = np.sqrt(mean_squared_error(y_train_inv, y_pred_train))
    r2_tr   = r2_score(y_train_inv, y_pred_train)

    metrics[commodity] = {
        'test':  {'MAE': mae_t,  'RMSE': rmse_t,  'R2': r2_t},
        'train': {'MAE': mae_tr, 'RMSE': rmse_tr, 'R2': r2_tr}
    }

    # Message résumé
    print(f"Résultats pour {commodity} :")
    print(f"  test -- MAE={mae_t:.4f}, RMSE={rmse_t:.4f}, R2={r2_t:.4f}")
    print(f"  train -- MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f}")

# Métriques globales
mae_t_g  = mean_absolute_error(global_y_true, global_y_pred)
rmse_t_g = np.sqrt(mean_squared_error(global_y_true, global_y_pred))
r2_t_g   = r2_score(global_y_true, global_y_pred)

# Ici on réutilise les derniers mae_tr, rmse_tr, r2_tr calculés
global_metrics = {
    'test':  {'MAE': mae_t_g,  'RMSE': rmse_t_g,  'R2': r2_t_g},
    'train': {'MAE': mae_tr,   'RMSE': rmse_tr,   'R2': r2_tr}
}

# Sauvegarde des fichiers
pd.DataFrame(results).to_csv("lstm_result.csv", index=False)              # date,commodity,y_true,y_pred
pd.DataFrame(future_predictions).to_csv("lstm_predict.csv", index=False)  # commodity,date,predicted

with open("lstm_metrics.json",        "w") as f: json.dump(metrics, f, indent=2)
with open("lstm_global_metrics.json", "w") as f: json.dump(global_metrics, f, indent=2)
