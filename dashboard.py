import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

# === CONFIGURATION ===
st.set_page_config(layout="wide")
st.title("Dashboard de prévision du prix des commodités")
st.markdown("Comparaison des modèles : XGBoost, CatBoost, LSTM, Prophet, GRU, Random Forest")

# === FICHIERS À CHARGER ===
models = {
    'XGBoost':   ('xgb_results.csv',      'xgb_global_metrics.json'),
    'CatBoost':  ('catboost_results.csv', 'catboost_global_metrics.json'),
    'LSTM':      ('lstm_results.csv',     'lstm_global_metrics.json'),
    'GRU':       ('gru_results.csv',      'gru_global_metrics.json'),
    'Random Forest': ('rf_results.csv', 'rf_global_metrics.json'),
    'PROPHET': ('stacking_results.csv', 'stacking_metrics.json'),
}

dfs = {}
metrics = {}

# === CHARGEMENT DES DONNÉES ===
for name, (csv_file, json_file) in models.items():
    try:
        dfs[name] = pd.read_csv(csv_file, parse_dates=['date'])
        with open(json_file, 'r') as f:
            metrics[name] = json.load(f)
    except FileNotFoundError:
        st.warning(f" Fichiers manquants pour le modèle {name}.")

# === AFFICHAGE DU TABLEAU DE MÉTRIQUES ===
if metrics:
    st.subheader("Résultats comparés (MAE, RMSE, R²)")

    # Reformater les données pour afficher les métriques d'entraînement et de test dans un format plus lisible
    results = []
    for model_name, metric_values in metrics.items():
        # Si les métriques d'entraînement ou de test sont manquantes, assigner une valeur par défaut
        train_mae = f"{metric_values.get('train', {}).get('MAE', 0.0):.2f}"
        train_rmse = f"{metric_values.get('train', {}).get('RMSE', 0.0):.2f}"
        train_r2 = f"{metric_values.get('train', {}).get('R2', 0.0):.4f}"
        
        test_mae = f"{metric_values.get('test', {}).get('MAE', 0.0):.2f}"
        test_rmse = f"{metric_values.get('test', {}).get('RMSE', 0.0):.2f}"
        test_r2 = f"{metric_values.get('test', {}).get('R2', 0.0):.4f}"
        
        results.append({
            'Modèle': model_name,
            'Train MAE': train_mae,
            'Train RMSE': train_rmse,
            'Train R²': train_r2,
            'Test MAE': test_mae,
            'Test RMSE': test_rmse,
            'Test R²': test_r2
        })

    table = pd.DataFrame(results)
    st.dataframe(table, use_container_width=True)
else:
    st.warning("Aucune métrique disponible.")

# === SÉLECTEUR DE MODÈLE ===
if dfs:
    model_choice = st.selectbox("Choisissez un modèle à visualiser :", list(dfs.keys()))
    df = dfs[model_choice]

    # === SÉLECTEUR DE COMMODITY SI DISPONIBLE ===
    selected_commodity = None
    if 'commodity' in df.columns:
        commodities = df['commodity'].unique()
        selected_commodity = st.selectbox("Choisissez la commodité :", commodities)
        df = df[df['commodity'] == selected_commodity]

    # === SÉLECTEUR DE PÉRIODE ===
    min_date = pd.to_datetime(df['date'].min()).to_pydatetime()
    max_date = pd.to_datetime(df['date'].max()).to_pydatetime()

    start, end = st.slider(
        "Filtrer la période de prédiction :",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    df_filtered = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

    # === AFFICHAGE DU GRAPHIQUE INTERACTIF ===
    fig = px.line(
        df_filtered, x='date', y=['y_true', 'y_pred'],
        labels={'value': 'Prix ($)', 'date': 'Date'},
        title=f"Prédiction vs Réalité — {model_choice}" + (f" ({selected_commodity})" if selected_commodity else "")
    )

    if len(fig.data) > 1:
        fig.data[1].line.color = 'red'  # courbe prédite en rouge

    fig.update_layout(legend_title_text='', legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Aucun fichier de données n'est disponible pour afficher les graphiques.")
