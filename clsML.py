import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report, r2_score, 
                             mean_absolute_error, roc_curve, auc, precision_recall_curve)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# 1. Loading & Data Preparation
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'Master_Raw_Data.csv')
df = pd.read_csv(csv_path)

# Feature Engineering เชิงพลวัต
df['pop_velocity'] = df.groupby('track_id')['track_popularity'].diff().fillna(0)
df['rank_change'] = df.groupby('track_id')['playlist_rank'].diff().fillna(0)

# จัดการข้อมูลแนวเพลง (คัดเฉพาะกลุ่ม Top 10 กระแสหลัก)
df['main_genre'] = df['artist_genres'].apply(lambda x: str(x).split(',')[0].strip())
top_genres = df['main_genre'].value_counts().nlargest(10).index
df['main_genre'] = df['main_genre'].apply(lambda x: x if x in top_genres else 'other')
df_encoded = pd.get_dummies(df, columns=['main_genre'], prefix='genre')

# คัดกรองเฉพาะเพลงที่มีจุดสังเกตต่อเนื่องเพียงพอ (ขั้นต่ำ 12 จุดเวลา หรือประมาณ 4 สัปดาห์)
track_counts = df_encoded['track_id'].value_counts()
valid_tracks = track_counts[track_counts >= 12].index
filtered_df = df_encoded[df_encoded['track_id'].isin(valid_tracks)].copy()

# คัดเลือกเกณฑ์ตัดสินสถานะเพลงฮิตด้วยค่ามัธยฐานของชุดข้อมูลจริง
median_threshold = filtered_df['track_popularity'].median()

# ทำการ Shift ตัวแปรตามพยากรณ์ไปข้างหน้า 6 ขั้นเวลา (ล่วงหน้า 14 วัน)
filtered_df['target_pop'] = filtered_df.groupby('track_id')['track_popularity'].shift(-6)
filtered_df['is_hit'] = (filtered_df['target_pop'] >= median_threshold).astype(int)
final_df = filtered_df.dropna(subset=['target_pop']).reset_index(drop=True)

# 2. Features Definition
features = (
    [f'playlist_rank_t-{i}' for i in range(6, 0, -1)] +
    [f'pop_velocity_t-{i}' for i in range(6, 0, -1)] +
    [f'rank_change_t-{i}' for i in range(6, 0, -1)] +
    ['artist_popularity', 'artist_followers'] +
    [col for col in final_df.columns if col.startswith('genre_')]
)

# [หมายเหตุ] 
# ก่อนรันโมเดลขั้นถัดไป ข้อมูลตัวแปรต้น (X) ต้องผ่านขั้นตอนการสไลด์หน้าต่างเวลา (Sliding Window) 
# เพื่อจับคู่ข้อมูลย้อนหลัง 6 จุดเวลาเข้าสู่อาเรย์สำหรับทำนายโมเดลต่อไป

# ตัวอย่างแนวทางการแบ่งส่วนชุดข้อมูลทดสอบมาตรฐานสัดส่วน 64:16:20
# X, y = final_df[features], final_df['is_hit']
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42)