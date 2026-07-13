import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ==========================================
# 1. DATA FLOW & PREPROCESSING PIPELINE
# ==========================================
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'Master_Raw_Data.csv')
df = pd.read_csv(csv_path)

# Enforce uniform genre confinement boundaries
df['main_genre'] = df['artist_genres'].apply(lambda x: str(x).split(',')[0].strip())
top_10 = df['main_genre'].value_counts().nlargest(10).index
df['main_genre'] = df['main_genre'].apply(lambda x: x if x in top_10 else 'other')
df_encoded = pd.get_dummies(df, columns=['main_genre'], prefix='genre')

# Extract dynamic delta attributes
df_encoded['pop_velocity'] = df_encoded.groupby('track_id')['track_popularity'].diff().fillna(0)
df_encoded['rank_change'] = df_encoded.groupby('track_id')['playlist_rank'].diff().fillna(0)

counts = df_encoded['track_id'].value_counts()
valid_ids = counts[counts >= 12].index
filtered_df = df_encoded[df_encoded['track_id'].isin(valid_ids)]
true_median = filtered_df['track_popularity'].median()

def create_windows(track_list, data, seq_len=6, horizon=6, threshold=56):
    """Compile raw continuous records into explicit sequential and static components."""
    all_seqs, all_statics, all_targets_cls, all_targets_reg = [], [], [], []
    dynamic_cols = ['playlist_rank', 'pop_velocity', 'rank_change'] 
    static_cols = ['artist_popularity', 'artist_followers'] + [c for c in data.columns if c.startswith('genre_')]
    for tid in track_list:
        group = data[data['track_id'] == tid]
        for i in range(len(group) - seq_len - horizon + 1):
            all_seqs.append(group[dynamic_cols].iloc[i : i + seq_len].values)
            all_statics.append(group[static_cols].iloc[i + seq_len - 1].values)
            
            # Target alignment: Forecast future popularity horizon (+6 steps = 14 days)
            target_val = group['track_popularity'].iloc[i + seq_len + horizon - 1]
            all_targets_cls.append(1 if target_val >= threshold else 0)
            all_targets_reg.append(target_val)
    return np.array(all_seqs), np.array(all_statics), np.array(all_targets_cls), np.array(all_targets_reg)

X_seq, X_sta, y_cls, y_reg = create_windows(valid_ids, filtered_df, threshold=true_median)

# Fit and apply standard scaling transformation parameters
scaler_seq, scaler_sta, scaler_y = StandardScaler(), StandardScaler(), StandardScaler()
X_seq_sc = scaler_seq.fit_transform(X_seq.reshape(-1, 3)).reshape(X_seq.shape)
X_sta_sc = scaler_sta.fit_transform(X_sta)
y_reg_sc = scaler_y.fit_transform(y_reg.reshape(-1, 1)).flatten()

# Split execution blocks ensuring stratified validation safety limits
X_tr_seq_c, X_te_seq_c, X_tr_sta_c, X_te_sta_c, y_tr_c, y_te_c = train_test_split(X_seq_sc, X_sta_sc, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
X_tr_seq_r, X_te_seq_r, X_tr_sta_r, X_te_sta_r, y_tr_r, y_te_r = train_test_split(X_seq_sc, X_sta_sc, y_reg_sc, test_size=0.2, random_state=42)

# ==========================================
# 2. PROPOSED DUAL-INPUT MODEL BUILDER
# ==========================================
def build_model(model_type, task, static_dim):
    """Constructs a Dual-Input Network Structure with Late Fusion Topology."""
    # Input definition blocks: Separating sequential time-series from static context features
    seq_in = Input(shape=(6, 3), name='Sequential_Dynamic_Input')
    sta_in = Input(shape=(static_dim,), name='Static_Profile_Input')
    
    # Branch 1: Temporal Context Branch (LSTM/GRU Sequence processing)
    x = (LSTM(64)(seq_in) if model_type == 'LSTM' else GRU(64)(seq_in))
    
    # Branch 2: Static Operational Branch
    y = Dense(32, activation='relu')(sta_in)
    
    # Late Fusion Interface Layer Layout
    merged = Concatenate()([Dropout(0.2)(x), y])
    z = Dense(32, activation='relu')(merged)
    out = Dense(1, activation='sigmoid')(z) if task == 'classification' else Dense(1, activation='linear')(z)
    
    m = Model(inputs=[seq_in, sta_in], outputs=out)
    m.compile(optimizer='adam', 
              loss='binary_crossentropy' if task == 'classification' else 'mse', 
              metrics=['accuracy' if task == 'classification' else 'mae'])
    return m

results_cls, results_reg, preds_store = [], [], {}
best_f1, best_r2, cls_win, reg_win = -1, -1, "", ""

# Iterative execution logic to compare deep models across different operational tasks
for m_type in ['LSTM', 'GRU']:
    for task in ['classification', 'regression']:
        print(f"\n🚀 Initiating Deep Learning Context: {m_type} working on {task}...")
        X_s, X_st, y_t = (X_tr_seq_c, X_tr_sta_c, y_tr_c) if task == 'classification' else (X_tr_seq_r, X_tr_sta_r, y_tr_r)
        X_ts, X_tst, y_ts = (X_te_seq_c, X_te_sta_c, y_te_c) if task == 'classification' else (X_te_seq_r, X_te_sta_r, y_te_r)
        
        model = build_model(m_type, task, X_st.shape[1])
        
        # Enforce EarlyStopping callback interface to safeguard system resource allocation
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit([X_s, X_st], y_t, validation_split=0.2, epochs=150, batch_size=32, callbacks=[es], verbose=1)
        
        p = model.predict([X_ts, X_tst])
        preds_store[f"{m_type}_{task}"] = (p, y_ts)
        
        if task == 'classification':
            y_p = (p > 0.5).astype(int)
            acc, prec, rec, f1 = accuracy_score(y_ts, y_p), precision_score(y_ts, y_p), recall_score(y_ts, y_p), f1_score(y_ts, y_p)
            results_cls.append({'Model': m_type, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
            if f1 > best_f1: best_f1, cls_win = f1, m_type
        else:
            r2 = r2_score(y_ts, p)
            results_reg.append({'Model': m_type, 'R-Squared': r2, 'MAE': mean_absolute_error(y_ts, p)})
            if r2 > best_r2: best_r2, reg_win = r2, m_type

# ==========================================
# 3. SUMMARY STATS MONITOR REPORT
# ==========================================
print("\n" + "="*85)
print("DEEP NEURAL SYSTEM METRICS REGISTRY REPORT")
print("="*85)
print(pd.DataFrame(results_cls).round(4).to_string(index=False))
print("\n" + "-"*85)
print(pd.DataFrame(results_reg).round(4).to_string(index=False))
