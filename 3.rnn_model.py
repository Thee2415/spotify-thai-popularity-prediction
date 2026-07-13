import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# 1. Load configuration and clean tabular vectors
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'Master_Raw_Data.csv')
df = pd.read_csv(csv_path)

# เตรียมโครงสร้างฟีเจอร์เชิงพลวัตและคัดแยกมิติแนวเพลงหลัก
df['main_genre'] = df['artist_genres'].apply(lambda x: str(x).split(',')[0].strip())
top_genres = df['main_genre'].value_counts().nlargest(10).index
df['main_genre'] = df['main_genre'].apply(lambda x: x if x in top_genres else 'other')

df_encoded = pd.get_dummies(df, columns=['main_genre'], prefix='genre')
df_encoded['pop_velocity'] = df_encoded.groupby('track_id')['track_popularity'].diff().fillna(0)
df_encoded['rank_change'] = df_encoded.groupby('track_id')['playlist_rank'].diff().fillna(0)

# คัดกรองความยาวซีเควนซ์ขั้นต่ำสำหรับการเรียนรู้แบบลำดับเวลา
track_counts = df_encoded['track_id'].value_counts()
valid_ids = track_counts[track_counts >= 12].index
filtered_df = df_encoded[df_encoded['track_id'].isin(valid_ids)]

median_threshold = filtered_df['track_popularity'].median()

# 2. Build Proposed Dual-Input Network Architecture
def create_dual_input_network(seq_shape, static_shape, rnn_type='lstm'):
    # เส้นทางที่ 1: การเรียนรู้ข้อมูลอนุกรมเวลา (Temporal Processing Branch)
    seq_input = Input(shape=seq_shape, name='sequential_input')
    if rnn_type.lower() == 'gru':
        rnn_out = GRU(64, dropout=0.2, recurrent_dropout=0.2)(seq_input)
    else:
        rnn_out = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(seq_input)
        
    # เส้นทางที่ 2: การเรียนรู้ข้อมูลคงที่เชิงบริบทศิลปิน (Static Processing Branch)
    static_input = Input(shape=static_shape, name='static_input')
    static_dense = Dense(32, activation='relu')(static_input)
    static_out = Dropout(0.2)(static_dense)
    
    # รวมข้อมูลจากทั้งสองเส้นทางเข้าด้วยกัน (Late Fusion Topology)
    combined = Concatenate()([rnn_out, static_out])
    shared_dense = Dense(32, activation='relu')(combined)
    shared_out = Dropout(0.2)(shared_dense)
    
    # ส่วนพยากรณ์ปลายทางแบบ Multi-task Learning Output Layers
    classification_output = Dense(1, activation='sigmoid', name='hit_output')(shared_out)
    regression_output = Dense(1, activation='linear', name='popularity_output')(shared_out)
    
    model = Model(inputs=[seq_input, static_input], outputs=[classification_output, regression_output])
    return model

# ตัวอย่างการตรวจสอบรายละเอียดสถาปัตยกรรมโครงข่ายประสาทเทียม
# nn_model = create_dual_input_network(seq_shape=(6, 3), static_shape=(11,), rnn_type='lstm')
# nn_model.summary()
