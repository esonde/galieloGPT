import streamlit as st
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, date

# Imposta la pagina: layout wide per desktop e responsive per dispositivi mobili
st.set_page_config(page_title="Elo 2v2 - Modello Avanzato", layout="wide")

###############################################
# COSTANTI DI CONFIGURAZIONE
###############################################
L = 5         # Numero di livelli per il morale (0,1,2,3,4)
L_spinta = 5  # Numero di livelli per la spinta (0,1,2,3,4)

###############################################
# FUNZIONI DI CARICAMENTO (cache)
###############################################
@st.cache_data(show_spinner=False)
def load_players():
    with open("players.json", "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_individual_vars():
    # individual_vars.csv contiene: player_id, Nome, Rating_offensivo, Apprendimento_offensivo, Rating_difensivo, Apprendimento_difensivo
    return pd.read_csv("individual_vars.csv")

@st.cache_data(show_spinner=False)
def load_global_vars():
    # global_vars.csv contiene: variable, value
    df = pd.read_csv("global_vars.csv")
    # Estraiamo M_vec e S_vec ordinati per indice
    M_vec = df[df['variable'].str.startswith("M")].sort_values("variable")['value'].to_numpy()
    S_vec = df[df['variable'].str.startswith("S")].sort_values("variable")['value'].to_numpy()
    return M_vec, S_vec

@st.cache_data(show_spinner=False)
def load_pair_vars():
    # pair_vars.csv contiene: player1_id, player1_Nome, player2_id, player2_Nome, Sinergia
    df = pd.read_csv("pair_vars.csv")
    # Costruiamo un dizionario per lookup: chiave = (player1_id, player2_id)
    syn_dict = { (row['player1_id'], row['player2_id']): row['Sinergia'] 
                for _, row in df.iterrows() }
    return syn_dict

@st.cache_data(show_spinner=False)
def load_match_history():
    with open("match.json", "r") as f:
        matches = json.load(f)
    df = pd.DataFrame(matches)
    # Converte la colonna Timestamp in datetime e ordina per tempo
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

###############################################
# CARICAMENTO DEI DATI
###############################################
players = load_players()
individual_vars = load_individual_vars()
M_vec, S_vec = load_global_vars()
syn_dict = load_pair_vars()
df_matches = load_match_history()

###############################################
# PREPARAZIONE DEI DATI DI STORICO: MORALE
###############################################
# Per ciascun ruolo definiamo Outcome: per Att1 e Dif1 è 1 (vittoria), per Att2 e Dif2 è 0.
df_long_morale = pd.DataFrame()
for role, outcome in [('Att1', 1.0), ('Dif1', 1.0), ('Att2', 0.0), ('Dif2', 0.0)]:
    temp = df_matches[[role]].copy()
    temp = temp.rename(columns={role: 'player'})
    temp['Outcome'] = outcome
    temp['match_idx'] = temp.index
    temp['role'] = role
    df_long_morale = pd.concat([df_long_morale, temp[['match_idx', 'player', 'role', 'Outcome']]], ignore_index=True)
df_long_morale.sort_values(['player', 'role', 'match_idx'], inplace=True)
# Per ogni giocatore e ruolo, calcoliamo la somma degli Outcome nelle 4 partite precedenti (shift=1 per escludere la partita corrente)
df_long_morale['win_count'] = df_long_morale.groupby(['player','role'])['Outcome'].shift(1).rolling(4, min_periods=1).sum()
df_long_morale['morale_idx'] = df_long_morale['win_count'].apply(lambda k: int(k) if (not pd.isna(k)) and k < L else L - 1)

def get_morale_index(pid, ruolo):
    """
    Per il ruolo 'attacker', considera sia Att1 che Att2.
    Per il ruolo 'defender', considera Dif1 e Dif2.
    Restituisce l'indice morale dell'ultima partita giocata.
    Se il giocatore non ha uno storico, restituisce 0.
    """
    if ruolo == "attacker":
        ruoli = ["Att1", "Att2"]
    else:
        ruoli = ["Dif1", "Dif2"]
    df_role = df_long_morale[(df_long_morale['player'] == pid) & (df_long_morale['role'].isin(ruoli))]
    if df_role.empty:
        return 0
    else:
        # Prendiamo l'ultima partita (in base a match_idx)
        last_row = df_role.iloc[-1]
        return int(last_row['morale_idx'])

###############################################
# PREPARAZIONE DEI DATI DI STORICO: SPINTA
###############################################
# Per spinta, raggruppiamo le partite per giorno (usando la data estratta dal Timestamp)
df_matches['day'] = df_matches['Timestamp'].dt.date
# Costruiamo un dataframe lungo con tutti i ruoli
df_long_spinta = pd.DataFrame()
for col in ['Att1', 'Att2', 'Dif1', 'Dif2']:
    temp = df_matches[['Timestamp', col]].copy()
    temp = temp.rename(columns={col: 'player'})
    temp['match_idx'] = temp.index
    temp['day'] = df_matches['day']
    df_long_spinta = pd.concat([df_long_spinta, temp[['match_idx', 'player', 'day']]], ignore_index=True)
df_long_spinta.sort_values(['player', 'day', 'match_idx'], inplace=True)
# Per ogni giocatore, in ogni giorno, la spinta è il numero di partite precedenti nello stesso giorno
df_long_spinta['global_spinta'] = df_long_spinta.groupby(['player', 'day']).cumcount()

def get_spinta_index(pid, prediction_day):
    """
    Restituisce il numero di partite giocate da pid nel giorno prediction_day.
    Il valore viene capato a L_spinta-1.
    """
    df_day = df_long_spinta[df_long_spinta['day'] == prediction_day]
    count = df_day[df_day['player'] == pid]['global_spinta'].max()
    if pd.isna(count):
        count = 0
    return int(min(count, L_spinta - 1))

###############################################
# Calcolo dei t (numero di partite per ruolo) dallo storico
###############################################
# Conta le partite come attaccante (unendo Att1 e Att2)
att_counts = df_matches['Att1'].value_counts().add(df_matches['Att2'].value_counts(), fill_value=0).to_dict()
# Conta le partite come difensore (unendo Dif1 e Dif2)
def_counts = df_matches['Dif1'].value_counts().add(df_matches['Dif2'].value_counts(), fill_value=0).to_dict()

###############################################
# PREPARAZIONE DELLA LISTA DEI GIOCATORI PER LA SELEZIONE
###############################################
player_df = individual_vars[['player_id', 'Nome']].drop_duplicates()
player_list = list(player_df.itertuples(index=False, name=None))
player_list = sorted(player_list, key=lambda x: x[1])

###############################################
# FUNZIONE PER CALCOLARE IL RATING EFFECTIVE DI UN GIOCATORE
###############################################
def get_effective_rating(pid, ruolo, prediction_day):
    """
    Per il ruolo "attacker" restituisce:
       base_offensivo + (Apprendimento_offensivo * t_att) + bonus
    Per il ruolo "defender" restituisce:
       base_difensivo + (Apprendimento_difensivo * t_def) + bonus
    Il bonus è dato da: M_vec[morale_idx] + S_vec[spinta_idx],
    dove morale_idx e spinta_idx vengono ricavati dinamicamente dallo storico.
    """
    df_player = individual_vars[individual_vars["player_id"] == pid]
    if df_player.empty:
        base = 1500.0
        appr = 0.0
    else:
        if ruolo == "attacker":
            base = df_player.iloc[0]["Rating_offensivo"]
            appr = df_player.iloc[0]["Apprendimento_offensivo"]
            t = att_counts.get(pid, 0)
        else:
            base = df_player.iloc[0]["Rating_difensivo"]
            appr = df_player.iloc[0]["Apprendimento_difensivo"]
            t = def_counts.get(pid, 0)
    morale_idx = get_morale_index(pid, ruolo)
    spinta_idx = get_spinta_index(pid, prediction_day)
    bonus = M_vec[morale_idx] + S_vec[spinta_idx]
    return base + appr * t + bonus

###############################################
# FUNZIONE PER OTTENERE IL PARAMETRO DI SINERGIA
###############################################
def get_sinergia(attacker_id, defender_id):
    # Il modello applica la sinergia alla coppia (attaccante, difensore)
    return syn_dict.get((attacker_id, defender_id), 0.0)

###############################################
# LAYOUT: TITOLO E SELEZIONE DEI GIOCATORI
###############################################
st.title("Predizione Partita 2v2 - Modello Avanzato")

st.markdown("**Seleziona 4 giocatori:** per ciascuna squadra scegli Attaccante e Difensore.")

col1, col2 = st.columns(2)
with col1:
    att1 = st.selectbox("Squadra 1 - Attaccante", player_list, key="att1", format_func=lambda x: x[1])
    dif1 = st.selectbox("Squadra 1 - Difensore", player_list, key="dif1", format_func=lambda x: x[1])
with col2:
    att2 = st.selectbox("Squadra 2 - Attaccante", player_list, key="att2", format_func=lambda x: x[1])
    dif2 = st.selectbox("Squadra 2 - Difensore", player_list, key="dif2", format_func=lambda x: x[1])

st.markdown("### Giocatori selezionati")
colA, colB = st.columns(2)
with colA:
    st.info(f"**Squadra 1:** {att1[1]} (ID: {att1[0]}) | {dif1[1]} (ID: {dif1[0]})")
with colB:
    st.info(f"**Squadra 2:** {att2[1]} (ID: {att2[0]}) | {dif2[1]} (ID: {dif2[0]})")

###############################################
# CALCOLO DELLA PREVISIONE CON IL MODELLO AVANZATO
###############################################
if st.button("Prevedi il risultato"):
    selected_ids = [att1[0], dif1[0], att2[0], dif2[0]]
    if len(set(selected_ids)) < 4:
        st.error("Errore: un giocatore non può appartenere a squadre diverse!")
    else:
        with st.spinner("Calcolo in corso..."):
            # Assumiamo che la partita prevista si giochi nella data odierna
            prediction_day = date.today()
            
            # Calcolo dei rating effective in base al ruolo e allo storico
            rating_att1 = get_effective_rating(att1[0], "attacker", prediction_day)
            rating_dif1 = get_effective_rating(dif1[0], "defender", prediction_day)
            rating_att2 = get_effective_rating(att2[0], "attacker", prediction_day)
            rating_dif2 = get_effective_rating(dif2[0], "defender", prediction_day)
            
            # Recupera la sinergia per le coppie in campo (applicata all'attaccante)
            sinergia_team1 = get_sinergia(att1[0], dif1[0])
            sinergia_team2 = get_sinergia(att2[0], dif2[0])
            
            # Il punteggio di ciascuna squadra è la somma del rating dell'attaccante (con sinergia) e del difensore
            team1_score = (rating_att1 + sinergia_team1) + rating_dif1
            team2_score = (rating_att2 + sinergia_team2) + rating_dif2
            
            # Calcola la probabilità di vittoria della Squadra 1 secondo la formula Elo
            prob_team1 = 1.0 / (1.0 + 10 ** ((team2_score - team1_score) / 400.0))
            
        st.success("Fatto!")
        st.metric(label="Probabilità di Vittoria Squadra 1", value=f"{prob_team1*100:.1f}%")
        if prob_team1 >= 0.5:
            st.balloons()
            st.markdown("<h2 style='color: green;'>Squadra 1 è favorita!</h2>", unsafe_allow_html=True)
        else:
            st.snow()
            st.markdown("<h2 style='color: red;'>Squadra 2 è favorita!</h2>", unsafe_allow_html=True)
