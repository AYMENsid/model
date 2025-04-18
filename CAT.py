import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import math

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df = df[df['commodity'] == 'Crude Oil'].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# === 2. Lissage (facultatif) ===
df['close'] = df['close'].rolling(window=3).mean()
df = df.dropna()

# === 3. Sélection des colonnes ===
features = ['open', 'high', 'low', 'volume']
target = 'close'

X = df[features]
y = df[target]

# === 4. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.35)

# === 5. Entraînement du modèle CatBoost ===
model = CatBoostRegressor(verbose=0)
model.fit(X_train, y_train)

# === 6. Prédiction ===
y_pred = model.predict(X_test)

# === 7. Évaluation ===
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nÉvaluation CatBoost :")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2  : {r2:.4f}")


# === Visualisation : Prédiction vs Réalité ===
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Prix réels', color='blue')
plt.plot(y_pred, label='Prix prédits', color='orange')
plt.title('📊 Prédiction des prix de clôture — CatBoost')
plt.xlabel('Jours')
plt.ylabel('Prix ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import json

# Récupérer les bonnes dates pour les prédictions
test_dates = df['date'].values[-len(y_test):]

# Sauvegarde des prédictions
df_cat = pd.DataFrame({
    'date': test_dates,
    'y_true': y_test.values,
    'y_pred': y_pred
})
df_cat.to_csv('catboost_results.csv', index=False)

# Sauvegarde des métriques
metrics_cat = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
with open('catboost_metrics.json','w') as f:
    json.dump(metrics_cat, f)
