# Spotify Thai Music Popularity Prediction
### Predicting Streaming Success Using Machine Learning and Dual-Input RNN Architectures

This repository contains the complete end-to-end source code for my final year research project at King Mongkut's University of Technology Thonburi (KMUTT). The core objective of this project is to build an automated system capable of Predicting track popularity scores and classifying "Hit" songs 14 days in advance (based on a 6-timestep temporal lookback).

---

## Project Context & Business Problem
In data science and business analytics, we often have to adapt to system constraints. When Spotify deprecated its public endpoint for song audio features (like energy and danceability), it broke traditional forecasting models. 

To solve this, I designed an alternative methodology: instead of relying on static audio signals, this system captures continuous **Dynamic Data (Streaming Momentum)** combined with **Metadata (Artist Profiles)**. This ensures the business can still discover trends and predict hit songs reliably without needing the deprecated audio data.

---

## Proposed Methodology & Architecture
The framework processes multi-dimensional datasets by separating time-series lookback sequences from general context variables. I evaluated two main modeling approaches to find the most accurate business solution:

1. **Classical Machine Learning (Flattened Vector Approach)**:
   * Logistic Regression (Baseline evaluation)
   * Random Forest
   * Extreme Gradient Boosting (XGBoost)
2. **Deep Learning Topology (Late Fusion Approach)**:
   * Dual-Input Long Short-Term Memory (LSTM) Neural Network
   * Dual-Input Gated Recurrent Unit (GRU) Neural Network

---

## Key Performance Metrics & Insights

### 1. Classification (Identifying Hit Status)
* **XGBoost** achieved the highest precision, accurately identifying hit tracks with a **98.08% Test Accuracy** and an **0.998 ROC AUC** score.

### 2. Regression (Predicting Exact Popularity Scores)
* **XGBoost** performed best overall, explaining data variance with an $R^2$ Score of **0.9692**.
* **LSTM** proved highly robust in capturing long-term sequential trends, achieving an $R^2$ of **0.9568**.
* **GRU** minimized individual errors effectively, delivering the lowest Mean Absolute Error (MAE) at **1.6121 points** on a 0-100 scale.

### 3. Core Takeaway
Through Model Explainability (SHAP Beeswarm diagnostics), the system isolated the top core operational indicators driving a track's success:
* **Artist Reputation Baseline (`artist_popularity` & `artist_followers`)**: A high baseline popularity accumulated by the artist serves as the strongest positive driver for future hits.
* **Market Trend Preference (`genre_thai hip hop`)**: The Thai Hip Hop genre shows a dominant positive impact on model decisions, highlighting strong current market consumer demand.
* **Playlist Visibility & Historical Momentum (`playlist_rank_t-1` & `rank_change` variables)**: Higher playlist placements at recent timesteps ($t-1$) and high positive ranking momentum at historical lookbacks ($t-5$, $t-6$) significantly catalyze a track's probability of becoming a commercial hit.

---

##  Project Repository Structure
* `import_data.py`: The automated daily ETL data pipeline connecting securely to the Spotify Web API.
* `cls_model.py`: Handles tabular feature engineering, implements the sliding window data transformation, and trains the classical ML models (including SHAP visual diagnostics).
* `rnn_model.py`: Assembles the deep learning network via Keras (TensorFlow), utilizing a multi-task learning structure to process static and sequential inputs simultaneously.

---
*Developed by Tree Thammano | Department of Mathematics (Statistics and Data Science), King Mongkut's University of Technology Thonburi (KMUTT)*
