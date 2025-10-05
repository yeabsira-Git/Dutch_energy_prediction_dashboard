import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
MODEL_FILENAME = 'lightgbm_demand_model.joblib'


# --- 1. UTILITY FUNCTIONS (Copied from Training Script) ---

# Helper function to sanitize names
def sanitize_feature_names(columns):
    new_cols = []
    for col in columns:
        col = str(col)
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        col = re.sub(r'^_+|_+$', '', col)
        col = re.sub(r'_{2,}', '_', col)
        new_cols.append(col)
    return new_cols

# Feature Engineering Function
def create_features(df):
    # This must be the first step so the temperature column name is correct for subsequent feature creation
    df.columns = sanitize_feature_names(df.columns) 
    
    df.index = pd.to_datetime(df.index)
    df['time_index'] = np.arange(len(df.index))
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Convert temperature to the sanitized name expected by the model
    temp_col_sanitized = sanitize_feature_names([TEMP_COL])[0]

    # ADDED ROBUST TEMPERATURE FEATURES
    df['temp_lag24'] = df[temp_col_sanitized].shift(24)
    df['temp_roll72'] = df[temp_col_sanitized].rolling(window=72, min_periods=1).mean().shift(1)
    df['temp_roll168'] = df[temp_col_sanitized].rolling(window=168, min_periods=1).mean().shift(1)

    return df

# Target Lag Function
def add_lags(df, target_col):
    df[f'{target_col}_lag24'] = df[target_col].shift(24)
    df[f'{target_col}_lag48'] = df[target_col].shift(48)
    df[f'{target_col}_roll72'] = df[target_col].shift(24).rolling(window=72).mean()
    return df

# --- 2. CACHING AND LOADING ---

@st.cache_resource
def load_model():
    """Load the trained LightGBM model from file."""
    try:
        model = joblib.load(MODEL_FILENAME)
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Please ensure it is in the same directory.")
        return None

# NEW FUNCTION: Loads raw data without caching, specifically for the temperature lookup
def load_raw_data(path):
    """Load the raw data for temperature lookup without caching/feature engineering."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading raw data for temperature lookup: {e}")
        return None

@st.cache_data
def load_data(path):
    """Load and preprocess the historical data."""
    try:
        df = pd.read_csv(path)
        # CRITICAL: This is where 'DateUTC' is set as index and thus removed from columns
        df.set_index(DATE_COL, inplace=True)
        df.index = pd.to_datetime(df.index)

        # --- Preprocessing steps from training script ---
        df = df[df.index <= '2025-06-30 23:00:00'].copy() # Only use historical actuals

        # Simple One-Hot Encoding/Categorical cleanup (to match the model's feature space)
        df_encoded = df.select_dtypes(exclude=['object', 'category'])

        df_historical_features = create_features(df_encoded.copy())
        df_historical_features = add_lags(df_historical_features, TARGET_COL).dropna()

        return df_historical_features
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        return None

# --- 3. STREAMLIT APP CORE ---

st.set_page_config(
    page_title="Dutch Energy Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💡 Early Energy Shortage Prediction Dashboard")
st.markdown("Forecasting hourly energy demand in Dutch neighborhoods using LightGBM.")

# --- Load Data and Model ---
# FIX 1: Corrected filename by removing the space
historical_df = load_data('cleaned_energy_weather_data(1).csv')
model = load_model()

if historical_df is not None and model is not None:
    # Get the last actual demand data for lag calculation
    last_actuals = historical_df[sanitize_feature_names([TARGET_COL])[0]]

    # --- Sidebar Configuration ---
    st.sidebar.header("Prediction Settings")

    # User selects the forecast start date
    max_forecast_date = datetime(2025, 12, 31, 23, 0, 0) # Maximum date from your training script

    forecast_start_date_input = st.sidebar.date_input(
        "Forecast Start Date (Day After Last Actual Data)",
        value=datetime(2025, 7, 1).date(), # Default to 2025-07-01 (as in your script)
        min_value=datetime(2025, 7, 1).date(),
        max_value=max_forecast_date.date()
    )

    # Convert date input to the required datetime object for indexing
    forecast_start_dt = pd.to_datetime(forecast_start_date_input).floor('D')

    # User selects the forecast horizon
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=60, value=7)

    # Calculate the full forecast end date (hourly data)
    forecast_end_dt = forecast_start_dt + pd.Timedelta(days=forecast_days) - pd.Timedelta(hours=1)

    st.sidebar.markdown("---")

    # --- Prediction Button ---
    if st.sidebar.button("Run Forecast", type="primary"):

        with st.spinner(f"Running recursive forecast for {forecast_days} days..."):

            # --- Prepare Future Data (Requires the full historical data again for temperature lookups) ---
            # FIX: Use the uncached load_raw_data to get the full data frame, including the 'DateUTC' column.
            df_raw = load_raw_data('cleaned_energy_weather_data(1).csv') 
            if df_raw is None: st.stop()

            # Now process the raw data for the temperature lookup
            df_raw.set_index(DATE_COL, inplace=True)
            df_raw.index = pd.to_datetime(df_raw.index)
            df_full_temp = df_raw

            # --- Recreate 2024 Climatology Forecast for Temperature (Exogenous