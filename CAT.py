import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# === 1. Charger les données ===
df = pd.read_csv('C:\\Users\\WINDOWS\\Downloads\\all_fuels_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['commodity', 'date'])

# === 2. Lissage par commodity ===
df['close'] = df.groupby('commodity')['close'].transform(lambda x: x.rolling(window=3).mean())
df = df.dropna()

# === 3. Encodage de la variable catégorielle ===
df['commodity'] = df['commodity'].astype('category')

# === 4. Définir les features ===
features = ['open', 'high', 'low', 'volume', 'commodity']
target = 'close'

# === 5. Split train/test stratifié par commodity ===
X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []
meta_train_list, meta_test_list = [], []

for commodity in df['commodity'].cat.categories:
    df_commodity = df[df['commodity'] == commodity].reset_index(drop=True)
    X_com = df_commodity[features]
    y_com = df_commodity[target]
    meta_com = df_commodity[['date', 'commodity']]

    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X_com, y_com, meta_com, test_size=0.35, shuffle=False
    )

    X_train_list.append(X_tr)
    X_test_list.append(X_te)
    y_train_list.append(y_tr)
    y_test_list.append(y_te)
    meta_train_list.append(m_tr)
    meta_test_list.append(m_te)

X_train = pd.concat(X_train_list).reset_index(drop=True)
X_test = pd.concat(X_test_list).reset_index(drop=True)
y_train = pd.concat(y_train_list).reset_index(drop=True)
y_test = pd.concat(y_test_list).reset_index(drop=True)
meta_test = pd.concat(meta_test_list).reset_index(drop=True)

# === 6. Entraînement du modèle CatBoost ===
model = CatBoostRegressor(verbose=0, cat_features=['commodity'])
model.fit(X_train, y_train)

# === 7. Prédictions ===
y_pred = model.predict(X_test)

# === 8. Évaluation globale ===
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModèle multi-commodities — Évaluation globale :")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# === 9. Sauvegarde des résultats ===
df_cat = pd.DataFrame({
    'date': meta_test['date'],
    'commodity': meta_test['commodity'],
    'y_true': y_test,
    'y_pred': y_pred
})
df_cat.to_csv('catboost_multi_commodities_results.csv', index=False)

metrics_cat = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
with open('catboost_multi_commodities_metrics.json','w') as f:
    json.dump(metrics_cat, f)

# === 10. Graphique par commodity ===
unique_commodities = df_cat['commodity'].unique()

for com in unique_commodities:
    subset = df_cat[df_cat['commodity'] == com]

    plt.figure(figsize=(12, 5))
    plt.plot(subset['date'], subset['y_true'], label='Prix réel', color='blue')
    plt.plot(subset['date'], subset['y_pred'], label='Prix prédit', color='orange')
    plt.title(f'Prédiction des prix — {com}')
    plt.xlabel('Date')
    plt.ylabel('Prix ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarde optionnelle de l’image
    
    plt.show()
