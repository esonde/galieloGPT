import streamlit as st
import json
import pandas as pd
import numpy as np
import math
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects

# Imposta una grafica moderna
st.set_page_config(page_title="Predizione Partita", layout="wide")


###############################################
# Definizione e registrazione del custom layer "CastLayer"
###############################################
class CastLayer(Layer):
    def __init__(self, target_dtype=tf.float16, **kwargs):
        super(CastLayer, self).__init__(**kwargs)
        self.target_dtype = tf.dtypes.as_dtype(target_dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super(CastLayer, self).get_config()
        config.update({"target_dtype": self.target_dtype.name})
        return config

    @classmethod
    def from_config(cls, config):
        if "dtype" in config and "target_dtype" not in config:
            config["target_dtype"] = config.pop("dtype")
        if "target_dtype" in config:
            config["target_dtype"] = tf.dtypes.as_dtype(config["target_dtype"])
        else:
            config["target_dtype"] = tf.float16
        return cls(**config)

get_custom_objects()["Cast"] = CastLayer

###############################################
# FUNZIONI DI AGGREGAZIONE DEI DATI STORICI
###############################################
def compute_player_stats(df):
    """
    Calcola statistiche cumulative per ogni giocatore basate sui match in ordine temporale.
    """
    stats = {}
    for idx, row in df.sort_values('Timestamp').iterrows():
        current_time = row['Timestamp']
        hour = current_time.hour + current_time.minute / 60.0
        if row['Pt1'] > row['Pt2']:
            winners = [row['Att1'], row['Dif1']]
            losers  = [row['Att2'], row['Dif2']]
            win_margin = 10 - row['Pt2']
        else:
            winners = [row['Att2'], row['Dif2']]
            losers  = [row['Att1'], row['Dif1']]
            win_margin = 10 - row['Pt1']
        for player in set([row['Att1'], row['Att2'], row['Dif1'], row['Dif2']]):
            if player not in stats:
                stats[player] = {
                    'matches': 0,
                    'wins': 0,
                    'cum_margin': 0.0,
                    'roll_wins': [],
                    'roll_margins': [],
                    'streak': 0,
                    'last_outcome': None,
                    'hours': []
                }
            stats[player]['matches'] += 1
            stats[player]['hours'].append(hour)
            if player in winners:
                stats[player]['wins'] += 1
                stats[player]['cum_margin'] += win_margin
                stats[player]['roll_wins'].append(1)
                stats[player]['roll_margins'].append(win_margin)
                if stats[player]['last_outcome'] == True:
                    stats[player]['streak'] += 1
                else:
                    stats[player]['streak'] = 1
                stats[player]['last_outcome'] = True
            else:
                if player in losers:
                    if row['Pt1'] > row['Pt2']:
                        margin = -(10 - row['Pt1'])
                    else:
                        margin = -(10 - row['Pt2'])
                else:
                    margin = 0
                stats[player]['cum_margin'] += margin
                stats[player]['roll_wins'].append(0)
                stats[player]['roll_margins'].append(margin)
                if stats[player]['last_outcome'] == False:
                    stats[player]['streak'] -= 1
                else:
                    stats[player]['streak'] = -1
                stats[player]['last_outcome'] = False
    for player, data in stats.items():
        if data['matches'] > 0:
            data['win_rate'] = data['wins'] / data['matches']
            data['avg_margin'] = data['cum_margin'] / data['matches']
            window = 3
            data['rolling_win_rate'] = np.mean(data['roll_wins'][-window:]) if data['roll_wins'] else 0.5
            data['rolling_margin'] = np.mean(data['roll_margins'][-window:]) if data['roll_margins'] else 0.0
            data['avg_hour'] = np.mean(data['hours'])
        else:
            data['win_rate'] = 0.5
            data['avg_margin'] = 0.0
            data['rolling_win_rate'] = 0.5
            data['rolling_margin'] = 0.0
            data['avg_hour'] = 12.0
    return stats

###############################################
# FUNZIONE PER AGGREGARE LE STATISTICHE DI DUE GIOCATORI (TEAM)
###############################################
def aggregate_team_stats(player1_stats, player2_stats):
    if player1_stats is None and player2_stats is None:
        return {}
    elif player1_stats is None:
        return player2_stats
    elif player2_stats is None:
        return player1_stats
    else:
        aggregated = {}
        aggregated['win_rate'] = (player1_stats.get('win_rate', 0.5) + player2_stats.get('win_rate', 0.5)) / 2
        aggregated['matches'] = player1_stats.get('matches', 0) + player2_stats.get('matches', 0)
        aggregated['avg_margin'] = (player1_stats.get('avg_margin', 0.0) + player2_stats.get('avg_margin', 0.0)) / 2
        aggregated['streak'] = (player1_stats.get('streak', 0) + player2_stats.get('streak', 0)) / 2
        aggregated['rolling_win_rate'] = (player1_stats.get('rolling_win_rate', 0.5) + player2_stats.get('rolling_win_rate', 0.5)) / 2
        aggregated['rolling_margin'] = (player1_stats.get('rolling_margin', 0.0) + player2_stats.get('rolling_margin', 0.0)) / 2
        aggregated['avg_hour'] = (player1_stats.get('avg_hour', 12.0) + player2_stats.get('avg_hour', 12.0)) / 2
        # Per il genere, media dei due (0=maschio, 1=femmina)
        gender1 = id_to_person.get(player1_stats.get('id', 0), {"Sesso": 0})["Sesso"]
        gender2 = id_to_person.get(player2_stats.get('id', 0), {"Sesso": 0})["Sesso"]
        aggregated['avg_gender'] = (gender1 + gender2) / 2
        return aggregated

###############################################
# FUNZIONI PER CALCOLARE LE FEATURE DELLA PARTITA
###############################################
def time_of_day_bin(hour):
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3

def compute_match_features(team1_stats, team2_stats, match_hour, day_of_week, month):
    """
    Data la statistica aggregata delle due squadre (ognuna composta da 2 giocatori)
    e i parametri temporali ipotetici, calcola le feature differenziali per il modello.
    """
    hour_val = match_hour
    sin_hour = math.sin(2 * math.pi * hour_val / 24)
    cos_hour = math.cos(2 * math.pi * hour_val / 24)
    match_norm = 0.5  # valore ipotetico
    def safe_stat(stats, key, default):
        return stats.get(key, default) if stats is not None else default

    t1 = team1_stats if team1_stats is not None else {}
    t2 = team2_stats if team2_stats is not None else {}

    win_rate_1 = safe_stat(t1, 'win_rate', 0.5)
    win_rate_2 = safe_stat(t2, 'win_rate', 0.5)
    exp_1 = safe_stat(t1, 'matches', 0)
    exp_2 = safe_stat(t2, 'matches', 0)
    avg_margin_1 = safe_stat(t1, 'avg_margin', 0.0)
    avg_margin_2 = safe_stat(t2, 'avg_margin', 0.0)
    streak_1 = safe_stat(t1, 'streak', 0)
    streak_2 = safe_stat(t2, 'streak', 0)
    roll_wr_1 = safe_stat(t1, 'rolling_win_rate', 0.5)
    roll_wr_2 = safe_stat(t2, 'rolling_win_rate', 0.5)
    roll_margin_1 = safe_stat(t1, 'rolling_margin', 0.0)
    roll_margin_2 = safe_stat(t2, 'rolling_margin', 0.0)
    avg_hour_1 = safe_stat(t1, 'avg_hour', 12.0)
    avg_hour_2 = safe_stat(t2, 'avg_hour', 12.0)

    diff_win_rate = win_rate_1 - win_rate_2
    diff_experience = exp_1 - exp_2
    total_experience = exp_1 + exp_2
    log_total_experience = math.log(total_experience + 1)
    diff_margin = avg_margin_1 - avg_margin_2
    diff_streak = streak_1 - streak_2
    diff_rolling_win_rate = roll_wr_1 - roll_wr_2
    diff_rolling_margin = roll_margin_1 - roll_margin_2
    typ_dev_1 = abs(match_hour - avg_hour_1)
    typ_dev_2 = abs(match_hour - avg_hour_2)
    diff_typical_dev = typ_dev_1 - typ_dev_2
    win_margin_interaction = diff_win_rate * diff_margin
    gender_1 = t1.get('avg_gender', 0)
    gender_2 = t2.get('avg_gender', 0)
    diff_gender = gender_1 - gender_2

    base_features = {
        'hour': hour_val,
        'sin_hour': sin_hour,
        'cos_hour': cos_hour,
        'day_of_week': day_of_week,
        'month': month,
        'match_norm': match_norm,
        'diff_win_rate': diff_win_rate,
        'diff_experience': diff_experience,
        'total_experience': total_experience,
        'log_total_experience': log_total_experience,
        'diff_margin': diff_margin,
        'diff_streak': diff_streak,
        'diff_rolling_win_rate': diff_rolling_win_rate,
        'diff_rolling_margin': diff_rolling_margin,
        'diff_typical_dev': diff_typical_dev,
        'win_margin_interaction': win_margin_interaction,
        'diff_pair_synergy': 0.0,
        'diff_gender': diff_gender
    }
    tod = time_of_day_bin(hour_val)
    dummy = {f"tod_{i}": 1 if i == tod else 0 for i in range(4)}
    features = {**base_features, **dummy}
    return features

###############################################
# CARICAMENTO DEI DATI E CALCOLO STATISTICHE
###############################################
@st.cache_data(show_spinner=False)
def load_data():
    with open('players.json', 'r') as f:
        players = json.load(f)
    with open('match.json', 'r') as f:
        matches = json.load(f)
    df = pd.DataFrame(matches)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.sort_values('Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['match_index'] = np.arange(len(df))
    stats = compute_player_stats(df)
    for key in stats.keys():
        stats[key]['id'] = int(key)
    return players, df, stats

players, df_raw, player_stats_dict = load_data()

# Definisci la variabile globale id_to_person (necessaria per recuperare il genere)
id_to_person = {int(k): {"Nome": v["Nome"].strip(), "Sesso": 1 if v["Sesso"].upper() == "F" else 0} 
                for k, v in players.items()}

st.session_state["players"] = players

player_list = sorted([(int(k), v["Nome"]) for k, v in players.items()], key=lambda x: x[1])

###############################################
# CARICAMENTO DEL MODELLO PRE-ADDESTRATO
###############################################
@st.cache_resource(show_spinner=False)
def load_trained_model():
    model = load_model("final_model.h5", compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={"cls_output": "binary_crossentropy", "reg_output": "mse"})
    return model

model = load_trained_model()

###############################################
# INTERFACCIA UTENTE CON STREAMLIT
###############################################
st.title("Predizione del Risultato di una Partita")
st.markdown(
    "Seleziona 4 giocatori: per ciascuna squadra, scegli un Attaccante e un Difensore."
)
st.sidebar.header("Seleziona i giocatori")

att1 = st.sidebar.selectbox("Attaccante Squadra 1", player_list, format_func=lambda x: x[1])
dif1 = st.sidebar.selectbox("Difensore Squadra 1", player_list, format_func=lambda x: x[1])
att2 = st.sidebar.selectbox("Attaccante Squadra 2", player_list, format_func=lambda x: x[1])
dif2 = st.sidebar.selectbox("Difensore Squadra 2", player_list, format_func=lambda x: x[1])

match_hour = st.sidebar.slider("Seleziona l'orario della partita (ora)", 0, 23, 20)
current_dt = datetime.now()
day_of_week = current_dt.weekday()
month = current_dt.month

st.markdown("### Giocatori selezionati")
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Squadra 1:** Attaccante: {att1[1]} (ID: {att1[0]}), Difensore: {dif1[1]} (ID: {dif1[0]})")
with col2:
    st.info(f"**Squadra 2:** Attaccante: {att2[1]} (ID: {att2[0]}), Difensore: {dif2[1]} (ID: {dif2[0]})")

if st.button("Prevedi il risultato della partita"):
    if att1[0] in [att2[0], dif2[0]] or dif1[0] in [att2[0], dif2[0]]:
        st.error("Un giocatore non può appartenere a squadre diverse!")
    else:
        with st.spinner("Calcolo delle feature e previsione in corso..."):
            # Recupera le statistiche per ciascun giocatore
            s_att1 = player_stats_dict.get(att1[0], None)
            s_dif1 = player_stats_dict.get(dif1[0], None)
            s_att2 = player_stats_dict.get(att2[0], None)
            s_dif2 = player_stats_dict.get(dif2[0], None)
            if s_att1 is not None: s_att1['id'] = att1[0]
            if s_dif1 is not None: s_dif1['id'] = dif1[0]
            if s_att2 is not None: s_att2['id'] = att2[0]
            if s_dif2 is not None: s_dif2['id'] = dif2[0]
            # Aggrega le statistiche per ciascuna squadra
            team1_stats = aggregate_team_stats(s_att1, s_dif1)
            team2_stats = aggregate_team_stats(s_att2, s_dif2)
            # Calcola le feature per il match
            features_dict = compute_match_features(team1_stats, team2_stats, match_hour, day_of_week, month)
            base_order = [
                'hour', 'sin_hour', 'cos_hour', 'day_of_week', 'month', 'match_norm',
                'diff_win_rate', 'diff_experience', 'total_experience', 'log_total_experience',
                'diff_margin', 'diff_streak', 'diff_rolling_win_rate', 'diff_rolling_margin',
                'diff_typical_dev', 'win_margin_interaction', 'diff_pair_synergy', 'diff_gender'
            ]
            dummy_order = [f"tod_{i}" for i in range(4)]
            numeric_feature_order = base_order + dummy_order
            df_features = pd.DataFrame([features_dict], columns=numeric_feature_order)
            
            # Prepara gli input per il modello: gli ID vengono presi dai rispettivi menu a tendina
            input_dict = {
                "Att1": np.array([att1[0]], dtype=np.int32),
                "Dif1": np.array([dif1[0]], dtype=np.int32),
                "Att2": np.array([att2[0]], dtype=np.int32),
                "Dif2": np.array([dif2[0]], dtype=np.int32),
                "numeric_features": np.array(df_features.values, dtype=np.float32)
            }
            
            preds = model.predict(input_dict)
            pred_cls = preds[0][0][0]  # probabilità che Squadra 1 vinca
            pred_reg = preds[1][0][0]  # punteggio previsto per il perdente
            
            st.success("Previsione completata!")
            colA, colB = st.columns(2)
            with colA:
                st.metric(label="Probabilità di Vittoria (Squadra 1)", value=f"{pred_cls * 100:.1f}%")
            with colB:
                st.metric(label="Punteggio Perdente Previsto", value=f"{pred_reg:.2f}")
            # Animazione: se prob. >= 50% usiamo balloons, altrimenti mostriamo un'emoji triste
            if pred_cls >= 0.5:
                st.balloons()
            else:
                st.markdown("<h1 style='color: red;'>😢</h1>", unsafe_allow_html=True)
