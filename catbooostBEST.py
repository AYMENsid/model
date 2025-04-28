import pandas as pd
import math
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
import os

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

metrics_per_com = {}
maes_train, rmses_train, r2s_train = [], [], []
maes_test, rmses_test, r2s_test = [], [], []

all_results = []

# === 6. Itération par commodity ===
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

    # === 7. Entraînement du modèle CatBoost ===
    model = CatBoostRegressor(verbose=0, cat_features=['commodity'])
    model.fit(X_tr, y_tr)

    # === 8. Prédictions ===
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    # === 9. Évaluation par commodity ===
    mae_train = mean_absolute_error(y_tr, y_train_pred)
    rmse_train = math.sqrt(mean_squared_error(y_tr, y_train_pred))
    r2_train = r2_score(y_tr, y_train_pred)
    maes_train.append(mae_train)
    rmses_train.append(rmse_train)
    r2s_train.append(r2_train)

    mae_test = mean_absolute_error(y_te, y_test_pred)
    rmse_test = math.sqrt(mean_squared_error(y_te, y_test_pred))
    r2_test = r2_score(y_te, y_test_pred)

    metrics_per_com[commodity] = {
        'train': {'MAE': mae_train, 'RMSE': rmse_train, 'R2': r2_train},
        'test': {'MAE': mae_test, 'RMSE': rmse_test, 'R2': r2_test}
    }

    maes_test.append(mae_test)
    rmses_test.append(rmse_test)
    r2s_test.append(r2_test)

    print(f"{commodity} -> Train MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
    print(f"{commodity} -> Test  MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}")

    # Sauvegarde des résultats
    df_out = pd.DataFrame({
        'date': m_te['date'],
        'commodity': commodity,
        'y_true': y_te.values,
        'y_pred': y_test_pred
    })
    all_results.append(df_out)

# === 10. Concaténation des résultats ===
df_results = pd.concat(all_results, ignore_index=True)
df_results.to_csv('catboost_results.csv', index=False)

# === 11. Calcul des métriques globales ===
global_metrics = {
    'train': {
        'MAE':  float(np.mean(maes_train)),
        'RMSE': float(np.mean(rmses_train)),
        'R2':   float(np.mean(r2s_train))
    },
    'test': {
        'MAE':  float(np.mean(maes_test)),
        'RMSE': float(np.mean(rmses_test)),
        'R2':   float(np.mean(r2s_test))
    }
}

# === 12. Sauvegarde des métriques détaillées et globales ===
with open('catboost_metrics.json', 'w') as f:
    json.dump({
        'by_commodity': metrics_per_com,
        'global': global_metrics
    }, f, indent=2)

# === 13. Sauvegarde des seules métriques globales ===
with open('catboost_global_metrics.json', 'w') as f:
    json.dump(global_metrics, f, indent=2)

# === 14. Affichage des métriques globales ===
print("\nMétriques globales (moyenne par commodity) :")
print(f"Train MAE : {global_metrics['train']['MAE']:.4f} | Train RMSE : {global_metrics['train']['RMSE']:.4f} | Train R2 : {global_metrics['train']['R2']:.4f}")
print(f"Test MAE  : {global_metrics['test']['MAE']:.4f} | Test RMSE  : {global_metrics['test']['RMSE']:.4f} | Test R2  : {global_metrics['test']['R2']:.4f}")
