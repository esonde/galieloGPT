import streamlit as st
import pandas as pd
import numpy as np
import math
import json

# Imposta la pagina: layout wide per desktop e responsive per dispositivi mobili
st.set_page_config(page_title="Elo 2v2", layout="wide")

###############################################
# Caricamento dei dati pre-salvati
###############################################
@st.cache_data(show_spinner=False)
def load_players():
    with open("players.json", "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_individual_vars():
    # File individual_vars.csv contiene: player_id, Nome, Rating_offensivo, Rating_difensivo
    return pd.read_csv("individual_vars.csv")

players = load_players()
individual_vars = load_individual_vars()

###############################################
# Preparazione della lista dei giocatori
###############################################
player_df = individual_vars[['player_id', 'Nome']].drop_duplicates()
player_list = list(player_df.itertuples(index=False, name=None))
player_list = sorted(player_list, key=lambda x: x[1])

###############################################
# Funzione per ottenere il rating (media off/dif)
###############################################
def get_rating(pid):
    df = individual_vars[individual_vars["player_id"] == pid]
    if not df.empty:
        return (df.iloc[0]["Rating_offensivo"] + df.iloc[0]["Rating_difensivo"]) / 2
    return 1500.0

###############################################
# Layout: Titolo e selezione giocatori (senza sidebar)
###############################################
st.title("Predizione Partita 2v2 - Sistema Elo")

st.markdown("**Seleziona 4 giocatori:** per ciascuna squadra scegli Attaccante e Difensore.")

# Usa colonne per organizzare i menu di selezione
col1, col2 = st.columns(2)
with col1:
    att1 = st.selectbox("Squadra 1 - Attaccante", player_list, key="att1", format_func=lambda x: x[1])
    dif1 = st.selectbox("Squadra 1 - Difensore", player_list, key="dif1", format_func=lambda x: x[1])
with col2:
    att2 = st.selectbox("Squadra 2 - Attaccante", player_list, key="att2", format_func=lambda x: x[1])
    dif2 = st.selectbox("Squadra 2 - Difensore", player_list, key="dif2", format_func=lambda x: x[1])

# Mostra in modo sintetico la selezione
st.markdown("### Giocatori selezionati")
colA, colB = st.columns(2)
with colA:
    st.info(f"**Squadra 1:** {att1[1]} (ID: {att1[0]}) | {dif1[1]} (ID: {dif1[0]})")
with colB:
    st.info(f"**Squadra 2:** {att2[1]} (ID: {att2[0]}) | {dif2[1]} (ID: {dif2[0]})")

###############################################
# Calcolo della previsione usando il sistema Elo
###############################################
if st.button("Prevedi il risultato"):
    selected_ids = [att1[0], dif1[0], att2[0], dif2[0]]
    if len(set(selected_ids)) < 4:
        st.error("Errore: un giocatore non può appartenere a squadre diverse!")
    else:
        with st.spinner("Calcolo in corso..."):
            rating_att1 = get_rating(att1[0])
            rating_dif1 = get_rating(dif1[0])
            rating_att2 = get_rating(att2[0])
            rating_dif2 = get_rating(dif2[0])
            
            team1_score = rating_att1 + rating_dif1
            team2_score = rating_att2 + rating_dif2
            
            prob_team1 = 1.0 / (1.0 + 10 ** ((team2_score - team1_score) / 400.0))
        
        st.success("Fatto!")
        st.metric(label="Probabilità di Vittoria Squadra 1", value=f"{prob_team1*100:.1f}%")
        if prob_team1 >= 0.5:
            st.balloons()
            st.markdown("<h2 style='color: green;'>Squadra 1 è favorita!</h2>", unsafe_allow_html=True)
        else:
            st.snow()
            st.markdown("<h2 style='color: red;'>Squadra 2 è favorita!</h2>", unsafe_allow_html=True)
