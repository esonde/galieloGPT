import streamlit as st
import pandas as pd
import numpy as np
import math
import json

# Imposta una grafica moderna
st.set_page_config(page_title="Predizione Partita - Sistema Elo", layout="wide")

###############################################
# Caricamento dei dati pre-salvati
###############################################
@st.cache_data(show_spinner=False)
def load_players():
    with open("players.json", "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_individual_vars():
    # Il file individual_vars.csv contiene: player_id, Nome, Rating_offensivo, Rating_difensivo
    return pd.read_csv("individual_vars.csv")

players = load_players()
individual_vars = load_individual_vars()

###############################################
# Preparazione della lista dei giocatori per la selectbox
###############################################
# Costruiamo la lista a partire dal DataFrame individual_vars
player_list = individual_vars[['player_id', 'Nome']].drop_duplicates()
player_list = list(player_list.itertuples(index=False, name=None))
player_list = sorted(player_list, key=lambda x: x[1])  # Ordina per Nome

###############################################
# Funzione per ottenere il rating di un giocatore
###############################################
def get_rating(pid):
    # Cerca il giocatore in individual_vars; se non lo trova, usa 1500 come default.
    df = individual_vars[individual_vars["player_id"] == pid]
    if not df.empty:
        # Usiamo la media tra il rating offensivo e il rating difensivo come valore rappresentativo
        return (df.iloc[0]["Rating_offensivo"] + df.iloc[0]["Rating_difensivo"]) / 2
    else:
        return 1500.0

###############################################
# INTERFACCIA UTENTE CON STREAMLIT
###############################################
st.title("Predizione del Risultato di una Partita (Sistema Elo)")

st.markdown("Seleziona 4 giocatori: per ciascuna squadra, scegli un Attaccante e un Difensore.")
st.sidebar.header("Seleziona i giocatori")

att1 = st.sidebar.selectbox("Attaccante Squadra 1", player_list, format_func=lambda x: x[1])
dif1 = st.sidebar.selectbox("Difensore Squadra 1", player_list, format_func=lambda x: x[1])
att2 = st.sidebar.selectbox("Attaccante Squadra 2", player_list, format_func=lambda x: x[1])
dif2 = st.sidebar.selectbox("Difensore Squadra 2", player_list, format_func=lambda x: x[1])

st.markdown("### Giocatori selezionati")
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Squadra 1:** Attaccante: {att1[1]} (ID: {att1[0]}), Difensore: {dif1[1]} (ID: {dif1[0]})")
with col2:
    st.info(f"**Squadra 2:** Attaccante: {att2[1]} (ID: {att2[0]}), Difensore: {dif2[1]} (ID: {dif2[0]})")

###############################################
# Calcolo della previsione usando il sistema Elo
###############################################
if st.button("Prevedi il risultato della partita"):
    # Controlla che non ci siano giocatori duplicati in squadre differenti
    selected_ids = [att1[0], dif1[0], att2[0], dif2[0]]
    if len(set(selected_ids)) < 4:
        st.error("Un giocatore non può appartenere a squadre diverse!")
    else:
        with st.spinner("Calcolo in corso..."):
            # Ottieni i rating per ciascun giocatore
            rating_att1 = get_rating(att1[0])
            rating_dif1 = get_rating(dif1[0])
            rating_att2 = get_rating(att2[0])
            rating_dif2 = get_rating(dif2[0])
            
            # Calcola lo score di ciascuna squadra
            team1_score = rating_att1 + rating_dif1
            team2_score = rating_att2 + rating_dif2
            
            # Calcola la probabilità che la Squadra 1 vinca
            prob_team1 = 1.0 / (1.0 + 10 ** ((team2_score - team1_score) / 400.0))
            
            st.success("Previsione completata!")
            st.metric(label="Probabilità di Vittoria (Squadra 1)", value=f"{prob_team1*100:.1f}%")
            
            # Animazione: se la probabilità è >= 50%, mostra i balloons, altrimenti un'emoji triste.
            if prob_team1 >= 0.5:
                st.balloons()
            else:
                st.markdown("<h1 style='color: red;'>😢</h1>", unsafe_allow_html=True)
