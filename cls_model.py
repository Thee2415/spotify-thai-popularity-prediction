import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import warnings
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report,
                             r2_score, mean_absolute_error, mean_squared_error,
                             roc_curve, auc, precision_recall_curve, average_precision_score)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# --- 1. DATA PREPROCESSING & FEATURE ENGINEERING ---
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, 'Master_Raw_Data.csv')
df = pd.read_csv(csv_path)

# Calculate dynamic tracking momentum features (Velocity & Rank Change)
df['pop_velocity'] = df.groupby('track_id')['track_popularity'].diff().fillna(0)
df['rank_change'] = df.groupby('track_id')['playlist_rank'].diff().fillna(0)

# Group artist genres into Top 10 + 'other' to reduce noise and optimize dimensions
df['main_genre'] = df['artist_genres'].apply(lambda x: str(x).split(',')[0].strip())
top_10 = df['main_genre'].value_counts().nlargest(10).index
df['main_genre'] = df['main_genre'].apply(lambda x: x if x in top_10 else 'other')
df_encoded = pd.get_dummies(df, columns=['main_genre'], prefix='genre')

# Filter out short-lived tracks; require at least 12 observations for steady sequential analysis
counts = df_encoded['track_id'].value_counts()
valid_track_ids = counts[counts >= 12].index
filtered_df = df_encoded[df_encoded['track_id'].isin(valid_track_ids)]
true_median = filtered_df['track_popularity'].median()

# Print business summary report
print("="*60)
print(f"DATA INTEGRITY & PERFORMANCE CONTROLS REPORT")
print("="*60)
print(f"✓ Valid Tracks (Observations >= 12): {len(valid_track_ids)}")
print(f"✓ Balanced Threshold Selection (Median): {true_median}")
print("="*60)

# --- 2. FLAT MATRIX SLIDING WINDOW LOGIC ---
def create_windows(track_list, data, seq_len=6, horizon=6, threshold=56):
    """Transforms continuous time-series rows into a structured lookback window matrix."""
    all_seqs, all_statics, all_targets_cls, all_targets_reg = [], [], [], []
    dynamic_cols = ['playlist_rank', 'pop_velocity', 'rank_change'] 
    static_cols = ['artist_popularity', 'artist_followers'] + [c for c in data.columns if c.startswith('genre_')]
    
    for tid in track_list:
        group = data[data['track_id'] == tid]
        for i in range(len(group) - seq_len - horizon + 1):
            all_seqs.append(group[dynamic_cols].iloc[i : i + seq_len].values)
            all_statics.append(group[static_cols].iloc[i + seq_len - 1].values)
            
            # Map the target variable to the future forecast horizon (+6 timesteps = 14 days)
            target_val = group['track_popularity'].iloc[i + seq_len + horizon - 1]
            all_targets_cls.append(1 if target_val >= threshold else 0)
            all_targets_reg.append(target_val)
                
    return np.array(all_seqs), np.array(all_statics), np.array(all_targets_cls), np.array(all_targets_reg)

X_seq, X_sta, y_cls, y_reg = create_windows(valid_track_ids, filtered_df, threshold=true_median)

# Normalize feature scales for optimal performance alignment
scaler_seq, scaler_sta = StandardScaler(), StandardScaler()
X_seq_sc = scaler_seq.fit_transform(X_seq.reshape(-1, 3)).reshape(X_seq.shape)
X_sta_sc = scaler_sta.fit_transform(X_sta)

# Concatenate flattened sequential columns with profile features into a 29-dimension array
X_bl = np.hstack([X_seq_sc.reshape(X_seq_sc.shape[0], -1), X_sta_sc])
feature_names = [f"{c}_t-{t}" for t in range(6, 0, -1) for c in ['playlist_rank', 'pop_velocity', 'rank_change']] + \
                ['artist_popularity', 'artist_followers'] + [c for c in filtered_df.columns if c.startswith('genre_')]

# --- 3. SPLITTING & RUNTIME EXECUTION ---
# Enforce a secure 64:16:20 split ratio using Stratified Distribution to prevent class bias
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_bl, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_bl, y_reg, test_size=0.2, random_state=42)

# --- 4. MODEL EVALUATION WORKFLOWS ---
cls_models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('XGBoost', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
]

cls_results = []
print("\n[Executing Classification Workflows...]")
for i, (name, model) in enumerate(cls_models):
    cv = cross_validate(model, X_train_c, y_train_c, cv=5, scoring='accuracy')
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)
    cls_results.append({
        'Model': name, 'CV_Acc_Mean': cv['test_score'].mean(), 'CV_Acc_Std': cv['test_score'].std(),
        'Test_Acc': accuracy_score(y_test_c, y_pred), 'Precision': precision_score(y_test_c, y_pred),
        'Recall': recall_score(y_test_c, y_pred), 'F1_Score': f1_score(y_test_c, y_pred)
    })

reg_models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]

reg_results = []
print("[Executing Regression Workflows...]")
for name, model in reg_models:
    model.fit(X_train_r, y_train_r)
    y_pred_reg = model.predict(X_test_r)
    reg_results.append({
        'Model': name, 'R2_Score': r2_score(y_test_r, y_pred_reg),
        'MAE': mean_absolute_error(y_test_r, y_pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
    })

# --- 5. MODEL INTERPRETATION (SHAP) ---
print("\n[Running SHAP Framework explainability diagnostics...]")
best_xgb_cls = cls_models[2][1]
explainer = shap.TreeExplainer(best_xgb_cls)
X_shap_test = X_test_c[:500] 
shap_values = explainer.shap_values(X_shap_test)

# Plotting SHAP Beeswarm to evaluate feature impact directions
plt.figure(figsize=(10, 8))
plt.title("SHAP Feature Impact Trajectory Direction")
shap.summary_plot(shap_values, X_shap_test, feature_names=feature_names)
plt.show()

# --- 6. PERFORMANCE SUMMARY REPORT MATRIX ---
print("\n" + "="*85)
print("CLASSICAL ML OPERATIONAL PERFORMANCE CONSOLE")
print("="*85)
print(pd.DataFrame(cls_results).round(4).to_string(index=False))
print("\n" + "-"*85)
print(pd.DataFrame(reg_results).round(4).to_string(index=False))
