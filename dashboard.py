import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

# === CONFIGURATION ===
st.set_page_config(layout="wide")
st.title("Dashboard de prévision du prix du pétrole brut")
st.markdown("Comparaison des modèles : XGBoost, CatBoost, LSTM, Prophet")

# === FICHIERS À CHARGER ===
models = {
    'XGBoost':   ('xgb_results.csv',      'xgb_metrics.json'),
    'CatBoost':  ('catboost_results.csv', 'catboost_metrics.json'),
    'LSTM':      ('lstm_results.csv',     'lstm_metrics.json'),
    'Prophet':   ('prophet_results.csv',  'prophet_metrics.json'),
    'GRU':   ('gru_results.csv', 'gru_metrics.json'),

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
    table = pd.DataFrame(metrics).T.rename_axis('Modèle').reset_index()
    st.dataframe(table.style.format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.4f}'}), use_container_width=True)
else:
    st.warning("Aucune métrique disponible.")

# === SÉLECTEUR DE MODÈLE ===
if dfs:
    model_choice = st.selectbox("Choisissez un modèle à visualiser :", list(dfs.keys()))
    df = dfs[model_choice]

    # === CONVERSION DES DATES POUR LE SLIDER ===
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
        title=f"Courbe de prédiction vs réelle — {model_choice}"
    )

    # Colorer la courbe de prédiction en rouge
    fig.data[1].line.color = 'red'
    fig.update_layout(legend_title_text='', legend=dict(orientation="h", y=1.1))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Aucun fichier de données n'est disponible pour afficher les graphiques.")
