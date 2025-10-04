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
DATA_PATH = 'merge_data_for_app.csv' # <-- FIXED: Corrected name (singular 'merge')
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)' # Using sanitized name for consistency

# The exact 50 features used for training (ORDER MATTERS!)
# The list is based on your final debug output.
FEATURES = [
    'Cov_ratio', 'Wind_Direction_degrees',
    'Hourly_Mean_Wind_Speed_0_1_m_s', 'Mean_Wind_Speed_0_1_m_s',
    'Maximum_Wind_Gust_0_1_m_s', 'Temperature_0_1_degrees_Celsius',
    'Dew_Point_Temperature_0_1_degrees_Celsius', 'Sunshine_Duration_0_1_hours',
    'Precipitation_Duration_0_1_hours', 'Hourly_Precipitation_Amount_0_1_mm',
    'Air_Pressure_0_1_hPa', 'Horizontal_Visibility', 'Cloud_Cover_octants',
    'Relative_Atmospheric_Humidity', 'Present_Weather_Code',
    'Present_Weather_Code_Indicator', 'Fog_0_no_1_yes', 'Rainfall_0_no_1_yes',
    'Snow_0_no_1_yes', 'Thunder_0_no_1_yes', 'Ice_Formation_0_no_1_yes',
    'Number_of_Stations', 'Global_Radiation_kW_m2',
    
    # One-Hot Encoded Features (These must be created by encode_categoricals)
    'CountryCode_NL', 'CreateDate_03_03_2025_12_24_13', 'CreateDate_04_09_2025_15_45',
    'UpdateDate_03_03_2025_12_24_13', 'UpdateDate_04_09_2025_15_45',
    'Time_of_Day_Day', 'Time_of_Day_Night', 'Detailed_Time_of_Day_Evening',
    'Detailed_Time_of_Day_Midnight', 'Detailed_Time_of_Day_Morning',
    'Detailed_Time_of_Day_Night', 'Detailed_Time_of_Day_Noon',
    
    # Time and Temperature Features
    'time_index', 'hour', 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'quarter',
    'is_weekend', 'temp_lag24', 'temp_roll72', 'temp_roll168',
    
    # Lag Features 
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

def encode_categoricals(df, max_categories=10):
    """Performs one-hot encoding on object columns (CRITICAL FIX)."""
    # Identify the original categorical columns that were encoded during training
    cols_to_encode = ['MeasureItem', 'CountryCode', 'CreateDate', 'UpdateDate', 'Time_of_Day', 'Detailed_Time_of_Day']
    
    # Ensure only columns that exist and are object/category types are encoded
    valid_cols_to_encode = [col for col in cols_to_encode if col in df.columns and df[col].dtype in ['object', 'category']]

    if valid_cols_to_encode:
        df = pd.get_dummies(df, columns=valid_cols_to_encode, dummy_na=False)
    
    df.columns = sanitize_feature_names(df.columns)
    return df

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
    
    # Use the consistent, sanitized column name for the temperature column
    df['temp_lag24'] = df[TEMP_COL].shift(24)
    df['temp_roll72'] = df[TEMP_COL].rolling(window=72, min_periods=1).mean().shift(1)
    df['temp_roll168'] = df[TEMP_COL].rolling(window=168, min_periods=1).mean().shift(1)
    
    return df

# --- 3. DATA LOADING AND VISUALIZATION PREP ---
# --- CRITICAL FIX: Encode Categorical Features ---
  
    """Loads and prepares the historical data, including CRITICAL categorical encoding."""
    try:
        df = pd.read_csv(DATA_PATH) 
        df = df.set_index(df.columns[0])
        df.index = pd.to_datetime(df.index)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}.")
        return pd.DataFrame() 

    # --- CRITICAL FIX: Encode Categorical Features ---
    df = encode_categoricals(df)
    df.columns = sanitize_feature_names(df.columns)
    # Use the consistent, sanitized column name for the temperature column
    df['Temperature_C'] = df[TEMP_COL] / 10
    df['Hour'] = df.index.hour
    df['Temperature_Category'] = np.where(df['Temperature_C'] < 0, 
                                          'Freezing Day', 'Above Freezing')
    
    return df

# --- 4. THE CORE RECURSIVE FORECAST FUNCTION ---
def generate_forecast(df_historical, n_hours=72):
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
    
    # Ensure temperature data is available for the full forecast horizon (n_hours)
    temp_24h = df_temp[TEMP_COL].tail(24).values
    reps = int(np.ceil(n_hours / 24))
    tiled_temp = np.tile(temp_24h, reps)[:n_hours]

    future_df[TEMP_COL] = tiled_temp
    
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
        
        # --- ROBUST PREDICTION CHECK ---
        X_current_features = X_current[FEATURES].copy()

        required_lag_features = [f'{TARGET_COL}_lag24', f'{TARGET_COL}_lag48', f'{TARGET_COL}_roll72']
        
        if not X_current_features[required_lag_features].isnull().values.any():
            prediction = model.predict(X_current_features)[0]
            future_df.loc[current_index, PREDICTION_COL_NAME] = prediction
        else:
            future_df.loc[current_index, PREDICTION_COL_NAME] = np.nan

    return future_df[[PREDICTION_COL_NAME]].dropna()
