import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import math

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === 2. Indicateurs temporels et techniques ===
df['close'] = df['close'].rolling(window=3).mean()  # Lissage
df['close_t-1'] = df['close'].shift(1)
df['close_t-2'] = df['close'].shift(2)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek

# Ajout de nouveaux indicateurs techniques
df['rsi'] = 100 - (100 / (1 + (df['close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / 
                           df['close'].diff().apply(lambda x: abs(min(x, 0))).rolling(window=14).mean())))
df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['bollinger_upper'] = df['close'].rolling(window=20).mean() + 2*df['close'].rolling(window=20).std()
df['bollinger_lower'] = df['close'].rolling(window=20).mean() - 2*df['close'].rolling(window=20).std()

# Supprimer les valeurs NaN dues au rolling/shift
df = df.dropna()

# === 3. Définir les features et la target ===
features = ['open', 'high', 'low', 'volume', 'close_t-1', 'close_t-2', 'sma_5', 'ema_10', 'month', 'dayofweek', 'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']
target = 'close'

X = df[features]
y = df[target]

# === 4. Séparation train/test chronologique ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.25)

# === 5. Recherche des meilleurs hyperparamètres avec GridSearch ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 6, 7],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Affichage des meilleurs hyperparamètres
print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")

# === 6. Modèle XGBoost avec les meilleurs paramètres ===
model = grid_search.best_estimator_
model.fit(X_train, y_train)

# === 7. Prédiction ===
y_pred = model.predict(X_test)

# === 8. Évaluation ===
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nÉvaluation XGBoost enrichi :")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# === 9. Visualisation ===
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Prix réels', color='blue')
plt.plot(y_pred, label='Prix prédits', color='green')
plt.title('📈 Prédiction des prix de clôture — XGBoost (avec indicateurs)')
plt.xlabel('Jours')
plt.ylabel('Prix ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import json

# Récupérer les dates correspondant à X_test (en utilisant le même slicing que y_test)
test_dates = df['date'].values[-len(y_test):]

# Créer le DataFrame des résultats
df_xgb = pd.DataFrame({
    'date': test_dates,
    'y_true': y_test.values,
    'y_pred': y_pred
})
df_xgb.to_csv('xgb_results.csv', index=False)

# Sauvegarder les métriques
metrics_xgb = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
with open('xgb_metrics.json', 'w') as f:
    json.dump(metrics_xgb, f)
