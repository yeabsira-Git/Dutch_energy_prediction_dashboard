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

# List of categorical columns identified from the model's expected features
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']
TARGET_COL_SANITIZED = 'Demand_MW' # Sanitize result for TARGET_COL

# --- 1. UTILITY FUNCTIONS (Copied from Training Script) ---

# Helper function to sanitize names
def sanitize_feature_names(columns):
    new_cols = []
    for col in columns:
        col = str(col)
        # Remove special characters and replace with underscore
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        # Clean up leading/trailing/multiple underscores
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
    if temp_col_sanitized in df.columns:
        df['temp_lag24'] = df[temp_col_sanitized].shift(24)
        df['temp_roll72'] = df[temp_col_sanitized].rolling(window=72, min_periods=1).mean().shift(1)
        df['temp_roll168'] = df[temp_col_sanitized].rolling(window=168, min_periods=1).mean().shift(1)
    else:
        pass 

    return df

# Target Lag Function (Used only for initial historical data setup)
def add_lags(df, target_col):
    target_col_sanitized = sanitize_feature_names([target_col])[0]
    df[target_col_sanitized] = df[target_col_sanitized] # Ensure sanitized column is used
    df[f'{target_col_sanitized}_lag24'] = df[target_col_sanitized].shift(24)
    df[f'{target_col_sanitized}_lag48'] = df[target_col_sanitized].shift(48)
    df[f'{target_col_sanitized}_roll72'] = df[target_col_sanitized].shift(24).rolling(window=72).mean()
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

# Loads raw data without caching, specifically for the temperature lookup
@st.cache_data
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
        df.set_index(DATE_COL, inplace=True)
        df.index = pd.to_datetime(df.index)

        # --- Preprocessing steps from training script ---
        df = df[df.index <= '2025-06-30 23:00:00'].copy() # Only use historical actuals

        # 1. PERFORM ONE-HOT ENCODING
        df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
        
        # 2. DROP ORIGINAL TEXT COLUMNS AND KEEP ONLY NUMERIC + ENCODED
        # Note: This is where we implicitly rely on pandas reading the weather data as numeric
        df_encoded = df_encoded.select_dtypes(exclude=['object', 'category'])
        
        # 3. CREATE TIME/LAG FEATURES
        df_historical_features = create_features(df_encoded.copy())
        df_historical_features = add_lags(df_historical_features, TARGET_COL).dropna()

        return df_historical_features
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        return None

# --- 3. CORE PREDICTION FUNCTION (MOVED AND CACHED) ---

# Function to run the heavy recursive forecast
def generate_forecast_results(historical_df, model, forecast_start_dt, forecast_end_dt, forecast_days):
    """Handles the full recursive prediction loop and data preparation."""
    
    # Constants from global scope
    # Note: These values rely on the global scope being set up correctly before calling this function.
    EXPECTED_FEATURES = model.feature_name_
    LAG_COLS = [f'{TARGET_COL_SANITIZED}_lag24', f'{TARGET_COL_SANITIZED}_lag48', f'{TARGET_COL_SANITIZED}_roll72']
    PREDICTION_COL_NAME = 'Predicted_Demand_MW'
    FORECAST_BEGIN_DT = pd.to_datetime('2025-07-01 00:00:00')

    # --- Prepare Future Data (Requires the full historical data again for exogenous lookups) ---
    df_raw = load_raw_data('cleaned_energy_weather_data(1).csv') 
    if df_raw is None: return None

    df_raw.set_index(DATE_COL, inplace=True)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_full_temp = df_raw.copy()

    # --- Recreate 2024 Climatology Forecast for ALL Exogenous Variables ---
    START_2024_CLIMATOLOGY = '2024-07-01 00:00:00'
    END_2024_CLIMATOLOGY = '2024-12-31 23:00:00'
    dates_2025_forecast_index = pd.date_range(start=datetime(2025, 7, 1), end=datetime(2025, 12, 31, 23, 0, 0), freq='h')
    
    EXOGENOUS_COLS_RAW = [col for col in df_full_temp.columns if col != TARGET_COL and col not in CATEGORICAL_COLS and col != DATE_COL]
    
    temp_history_climatology = df_full_temp.loc[START_2024_CLIMATOLOGY:END_2024_CLIMATOLOGY, EXOGENOUS_COLS_RAW].copy()

    if len(temp_history_climatology) != len(dates_2025_forecast_index):
        st.error("Climatology data length mismatch. Cannot proceed.") 
        return None
    
    future_exog_climatology_df = pd.DataFrame(
        temp_history_climatology.values, 
        index=dates_2025_forecast_index, 
        columns=temp_history_climatology.columns
    )
    future_exog_climatology_df[TARGET_COL] = np.nan 

    # --- Add Categorical Placeholders ---
    for col in CATEGORICAL_COLS:
        if col == 'Time_of_Day':
            future_exog_climatology_df[col] = np.where(future_exog_climatology_df.index.hour.isin(range(6, 18)), 'Day', 'Night')
        elif col == 'Detailed_Time_of_Day':
            future_exog_climatology_df[col] = future_exog_climatology_df.index.hour.map(lambda h: 
                'Morning' if h in range(5, 12) else 
                'Noon' if h in range(12, 17) else 
                'Evening' if h in range(17, 22) else 
                'Night'
            ).fillna('Night') 
        elif col == 'MeasureItem':
            future_exog_climatology_df[col] = 'Monthly Hourly Load Values'
        elif col == 'CountryCode':
            future_exog_climatology_df[col] = 'NL'
        elif col in ['CreateDate', 'UpdateDate']:
            future_exog_climatology_df[col] = '03-03-2025 12:24:13' 

    future_exog_df_encoded = pd.get_dummies(future_exog_climatology_df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    future_exog_df_encoded = create_features(future_exog_df_encoded.copy())
    
    # --- Combine and Final Alignment ---

    df_combined = pd.concat([historical_df[[TARGET_COL_SANITIZED]], future_exog_df_encoded], axis=0)
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

    idx_full_forecast = pd.date_range(start=FORECAST_BEGIN_DT, end=forecast_end_dt, freq='h')
    df_temp = df_combined.loc[idx_full_forecast].copy()

    future_df = pd.DataFrame(0, index=idx_full_forecast, columns=EXPECTED_FEATURES)

    for col in df_temp.columns:
        if col in future_df.columns:
            future_df[col] = df_temp[col]

    for col in EXPECTED_FEATURES:
        if col in future_df.columns:
            future_df[col] = pd.to_numeric(future_df[col], errors='coerce').astype(float)
            
    # --- Recursive Prediction Loop ---

    last_actuals = historical_df[TARGET_COL_SANITIZED].tail(72).copy()
    s_predictions = pd.Series(index=idx_full_forecast, dtype=float)
    
    # Use st.info instead of st.write inside the loop (faster)
    st.info(f"Calculating full recursive path from **{FORECAST_BEGIN_DT}** to **{forecast_end_dt}**...") 

    # IMPORTANT: Removed st.write from inside the loop to speed up rendering
    for i, current_index in enumerate(idx_full_forecast):
        X_current = future_df.loc[[current_index]].copy()
        s_latest = pd.concat([last_actuals, s_predictions.dropna()]).sort_index()

        # --- ROBUST LAG CALCULATION ---
        X_current.at[current_index, LAG_COLS[0]] = s_latest.get(current_index - pd.Timedelta(hours=24))
        X_current.at[current_index, LAG_COLS[1]] = s_latest.get(current_index - pd.Timedelta(hours=48))
        
        roll72_window_end = current_index - pd.Timedelta(hours=24)
        roll72_window_start = roll72_window_end - pd.Timedelta(hours=72)
        
        if s_latest.loc[roll72_window_start:roll72_window_end].empty:
            roll72_val = np.nan
        else:
            roll72_val = s_latest.loc[roll72_window_start:roll72_window_end].mean()
            
        X_current.at[current_index, LAG_COLS[2]] = roll72_val
        # ------------------------------

        if not np.isnan(X_current.at[current_index, LAG_COLS[2]]):
            prediction = model.predict(X_current[EXPECTED_FEATURES])[0]
        else:
            prediction = np.nan

        s_predictions.loc[current_index] = prediction

    # --- Finalize and Filter Results ---
    future_df[PREDICTION_COL_NAME] = s_predictions
    
    df_display_window = future_df.loc[forecast_start_dt:forecast_end_dt].copy()
    df_plot = df_display_window.dropna(subset=[PREDICTION_COL_NAME]) 
    
    # Calculate 90th percentile threshold
    shortage_threshold = historical_df[TARGET_COL_SANITIZED].quantile(0.90) 

    if df_plot.empty:
        st.error(f"Cannot display results. Please ensure your start date is at least 72 hours after {FORECAST_BEGIN_DT.date()}.")
        return None

    # Return all necessary results
    return {
        'df_plot': df_plot,
        'shortage_threshold': shortage_threshold,
        'PREDICTION_COL_NAME': PREDICTION_COL_NAME,
        'EXPECTED_FEATURES': EXPECTED_FEATURES,
        'forecast_start_dt': forecast_start_dt,
        'forecast_end_dt': forecast_end_dt,
        'forecast_days': forecast_days
    }


# --- 4. STREAMLIT APP CORE ---

st.set_page_config(
    page_title="Dutch Energy Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💡 Early Energy Shortage Prediction Dashboard")
st.markdown("Forecasting hourly energy demand in Dutch neighborhoods using LightGBM.")

# Initialize session state for caching results
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = {}

# --- Load Data and Model ---
historical_df = load_data('cleaned_energy_weather_data(1).csv')
model = load_model()

if historical_df is not None and model is not None:
    # Get the last actual demand data for lag calculation
    EXPECTED_FEATURES = model.feature_name_
    LAG_COLS = [f'{TARGET_COL_SANITIZED}_lag24', f'{TARGET_COL_SANITIZED}_lag48', f'{TARGET_COL_SANITIZED}_roll72']

    # --- Sidebar Configuration ---
    st.sidebar.header("Prediction Settings")

    # User selects the forecast start date
    max_forecast_date = datetime(2025, 12, 31, 23, 0, 0) 

    forecast_start_date_input = st.sidebar.date_input(
        "Forecast Start Date (Day After Last Actual Data)",
        value=datetime(2025, 7, 1).date(), 
        min_value=datetime(2025, 7, 1).date(),
        max_value=max_forecast_date.date(),
        key='forecast_date_input'
    )

    # Convert date input to the required datetime object for indexing
    forecast_start_dt = pd.to_datetime(forecast_start_date_input).floor('D')

    # User selects the forecast horizon (REDUCED MAX from 60 to 30 for faster runtimes)
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7, key='forecast_days_slider')

    # Calculate the full forecast end date (hourly data)
    forecast_end_dt = forecast_start_dt + pd.Timedelta(days=forecast_days) - pd.Timedelta(hours=1)
    
    st.sidebar.markdown("---")

    # Define current parameters for caching check
    current_params = {
        'start_dt': forecast_start_dt,
        'days': forecast_days
    }
    
    # --- Prediction Button ---
    # The button triggers the computation and updates session state
    if st.sidebar.button("Run Forecast", type="primary"):
        with st.spinner(f"Running recursive forecast for {forecast_days} days..."):
            results = generate_forecast_results(historical_df, model, forecast_start_dt, forecast_end_dt, forecast_days)
            if results:
                st.session_state.forecast_data = results
                st.session_state.last_run_params = current_params
                st.success(f"Forecast Complete! Results for **{forecast_start_dt.date()}** to **{forecast_end_dt.date()}** stored in cache.")
            else:
                st.error("Forecast failed to generate results.")
    
    # --- Display Forecast Results (ALWAYS runs if data is available in state) ---
    if st.session_state.forecast_data:
        
        # Unpack results from session state
        results = st.session_state.forecast_data
        df_plot = results['df_plot']
        shortage_threshold = results['shortage_threshold']
        PREDICTION_COL_NAME = results['PREDICTION_COL_NAME']
        EXPECTED_FEATURES = results['EXPECTED_FEATURES']
        
        # Use cached parameters for display headings
        start_date = st.session_state.last_run_params['start_dt'].date()
        end_date = (st.session_state.last_run_params['start_dt'] + pd.Timedelta(days=st.session_state.last_run_params['days']) - pd.Timedelta(hours=1)).date()

        st.subheader(f"Forecast Results ({start_date} to {end_date})")

        # Shortage analysis (Uses 90th percentile)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Peak Predicted Demand", value=f"{df_plot[PREDICTION_COL_NAME].max():,.2f} MW")
        with col2:
            st.metric(label="Shortage Threshold (90th Percentile)", value=f"{shortage_threshold:,.2f} MW")

        shortage_hours = df_plot[df_plot[PREDICTION_COL_NAME] > shortage_threshold]

        if not shortage_hours.empty:
            st.warning(f"🚨 **HIGH DEMAND ALERT:** Predicted demand exceeds the 90th percentile threshold during **{len(shortage_hours)} hours** in the forecast period. This requires proactive planning.")
            st.dataframe(shortage_hours.sort_values(PREDICTION_COL_NAME, ascending=False).head(), use_container_width=True)
        else:
            st.info("Predicted demand is below the 90th percentile threshold for high-stress events.")


        # Plotting the forecast
        st.subheader("Hourly Demand Forecast")
        st.line_chart(df_plot, y=PREDICTION_COL_NAME)

        st.subheader("Raw Data Preview")
        st.dataframe(df_plot[[PREDICTION_COL_NAME] + [f for f in EXPECTED_FEATURES if f in df_plot.columns]].head(10), use_container_width=True)

    # --- 5. Historical Demand Variance Analysis (New Section) ---
    st.subheader("📊 Historical Demand Variance by Time of Day and Temperature")
    st.markdown("Analyze how the historical average energy demand (MW) changes throughout the day under different temperature conditions.")

    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]

    # Function to categorize temperature (assuming temperature is in 0.1 degree Celsius, so divide by 10)
    def categorize_temp(temp):
        temp_c = temp / 10.0
        if temp_c < 5:
            return 'Cold (< 5°C)'
        elif temp_c < 15:
            return 'Mild (5°C - 15°C)'
        elif temp_c < 25:
            return 'Warm (15°C - 25°C)'
        else:
            return 'Hot (> 25°C)'

    # Create a copy of historical data to avoid modifying cached data
    df_analysis = historical_df.copy()

    # Apply categorization
    if TEMP_COL_SAN in df_analysis.columns and 'hour' in df_analysis.columns:
        df_analysis['Temp_Category'] = df_analysis[TEMP_COL_SAN].apply(categorize_temp)

        # Group by hour and temperature category, then calculate mean demand
        df_variance = df_analysis.groupby(['hour', 'Temp_Category'])[TARGET_COL_SANITIZED].mean().unstack()

        # Display the line chart
        if not df_variance.empty:
            st.line_chart(df_variance, use_container_width=True)
            st.caption("Each line represents the average demand across all historical data for the corresponding temperature band, showing variance over 24 hours.")
        else:
            st.warning("Historical analysis data is empty or missing required columns.")
    else:
        st.error(f"Temperature column ('{TEMP_COL_SAN}') or 'hour' column not found in the historical data for variance analysis.")
