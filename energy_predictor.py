# energy_predictor.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib 
import re
import warnings
warnings.filterwarnings("ignore") # Suppress minor warnings

# --- 1. CONFIGURATION AND CONSTANTS ---
MODEL_PATH = 'lgbm_model.pkl'
DATA_PATH = 'merge_data_for_app.csv' # <--- FIXED: Now points to your actual data file name
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature_0_1_degrees_Celsius'

# Define the features list (must match the model training features)
FEATURES = [
    'time_index', 'hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'quarter', 
    'is_weekend', 'temp_lag24', 'temp_roll72', 'temp_roll168', 
    f'{TARGET_COL}_lag24', f'{TARGET_COL}_lag48', f'{TARGET_COL}_roll72'
]

# --- 2. FEATURE ENGINEERING FUNCTIONS ---
def sanitize_feature_names(columns):
    """Aggressively cleans feature names for LightGBM."""
    new_cols = []
    for col in columns:
        col = str(col)
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        col = re.sub(r'^_+|_+$', '', col)
        col = re.sub(r'_{2,}', '_', col)
        new_cols.append(col)
    return new_cols

def create_features(df):
    """Creates time-based and external features."""
    df.index = pd.to_datetime(df.index)

    df['time_index'] = np.arange(len(df.index))
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    df['temp_lag24'] = df[TEMP_COL].shift(24)
    df['temp_roll72'] = df[TEMP_COL].rolling(window=72, min_periods=1).mean().shift(1)
    df['temp_roll168'] = df[TEMP_COL].rolling(window=168, min_periods=1).mean().shift(1)
    
    return df

# --- 3. DATA LOADING AND VISUALIZATION PREP ---
def load_and_preprocess_data():
    """Loads the final historical data for visualization."""
    try:
        # Tries to load the file named 'merge_data_for_app.csv'
        df = pd.read_csv(DATA_PATH) 
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        # This is the error you are seeing
        print(f"Error: Data file not found at {DATA_PATH}.")
        return pd.DataFrame() 

    df['Temperature_C'] = df[TEMP_COL] / 10
    df['Hour'] = df.index.hour
    df['Temperature_Category'] = np.where(df['Temperature_C'] < 0, 
                                        'Freezing Day', 'Above Freezing')
    
    return df

# --- 4. THE CORE RECURSIVE FORECAST FUNCTION ---
def generate_forecast(df_historical, n_hours=72, temp_climatology_path='energy_historical_features.csv'):
    """Generates the future forecast by implementing the recursive prediction."""
    PREDICTION_COL_NAME = 'Predicted_Demand_MW'
    
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model file not found. Returning empty forecast.")
        return pd.DataFrame() 

    # --- Setup Future Index and Exogenous Variables ---
    future_start = df_historical.index.max() + pd.Timedelta(hours=1)
    future_end = future_start + pd.Timedelta(hours=n_hours - 1)
    future_index = pd.date_range(start=future_start, end=future_end, freq='H')

    df_temp = df_historical[[TEMP_COL]].copy() 
    future_df = pd.DataFrame(index=future_index)
    
    future_df[TEMP_COL] = df_temp.iloc[-24:][TEMP_COL].values[:n_hours] 
    future_df[TARGET_COL] = np.nan
    
    df_full_extended = pd.concat([df_historical, future_df], axis=0)
    df_full_extended = create_features(df_full_extended)
    
    future_df = df_full_extended.loc[future_index].copy()
    future_df[PREDICTION_COL_NAME] = np.nan
    
    last_actuals = df_historical[TARGET_COL].tail(72).copy() 
    
    # === RECURSIVE PREDICTION LOOP ===
    for current_index in future_df.index:
        X_current = future_df.loc[[current_index]].copy()
        s_latest = pd.concat([last_actuals, future_df[PREDICTION_COL_NAME].dropna()]).sort_index()

        # --- LAG CALCULATION ---
        X_current.loc[current_index, f'{TARGET_COL}_lag24'] = s_latest.get(current_index - pd.Timedelta(hours=24))
        X_current.loc[current_index, f'{TARGET_COL}_lag48'] = s_latest.get(current_index - pd.Timedelta(hours=48))
        roll72_window_end = current_index - pd.Timedelta(hours=24)
        roll72_window_start = roll72_window_end - pd.Timedelta(hours=72)
        roll72_val = s_latest.loc[roll72_window_start:roll72_window_end].mean()
        X_current.loc[current_index, f'{TARGET_COL}_roll72'] = roll72_val
        
        X_current_features = X_current[FEATURES].copy()
        
        if not X_current_features[[f'{TARGET_COL}_lag24', f'{TARGET_COL}_lag48', f'{TARGET_COL}_roll72']].isnull().any(axis=1).all():
            prediction = model.predict(X_current_features)[0]
            future_df.loc[current_index, PREDICTION_COL_NAME] = prediction
        else:
            future_df.loc[current_index, PREDICTION_COL_NAME] = np.nan

    return future_df[[PREDICTION_COL_NAME]].dropna()
