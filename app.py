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
        st.error(f"Missing temperature column: {temp_col_sanitized}")

    return df

# Target Lag Function (Used only for initial historical data setup)
def add_lags(df, target_col):
    target_col_sanitized = sanitize_feature_names([target_col])[0]
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
        df_encoded = df_encoded.select_dtypes(exclude=['object', 'category'])
        
        # 3. CREATE TIME/LAG FEATURES
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
historical_df = load_data('cleaned_energy_weather_data(1).csv')
model = load_model()

if historical_df is not None and model is not None:
    # Get the last actual demand data for lag calculation
    TARGET_COL_SANITIZED = sanitize_feature_names([TARGET_COL])[0]
    last_actuals = historical_df[TARGET_COL_SANITIZED]

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
            df_raw = load_raw_data('cleaned_energy_weather_data(1).csv') 
            if df_raw is None: st.stop()

            # Now process the raw data for the temperature lookup
            df_raw.set_index(DATE_COL, inplace=True)
            df_raw.index = pd.to_datetime(df_raw.index)
            df_full_temp = df_raw

            # --- Recreate 2024 Climatology Forecast for Temperature (Exogenous Variable) ---
            START_2024_CLIMATOLOGY = '2024-07-01 00:00:00'
            END_2024_CLIMATOLOGY = '2024-12-31 23:00:00'

            temp_history_climatology = df_full_temp.loc[START_2024_CLIMATOLOGY:END_2024_CLIMATOLOGY, TEMP_COL].copy()

            # Create the future index that matches the forecast period
            dates_2025_forecast_index = pd.date_range(start=datetime(2025, 7, 1), end=datetime(2025, 12, 31, 23, 0, 0), freq='h')

            if len(temp_history_climatology) != len(dates_2025_forecast_index):
                st.error("Climatology data length mismatch. Cannot proceed. Check data consistency for 2024-07 to 2024-12.") 
                st.stop()
            
            temp_forecast = pd.Series(temp_history_climatology.values, index=dates_2025_forecast_index)
            temp_forecast.name = TEMP_COL

            future_exog_df = temp_forecast.to_frame()
            future_exog_df[TARGET_COL] = np.nan

            # --- CRITICAL FIX: Add and encode the required categorical columns for the future data ---
            
            # Replicate the values that created the exact dummy columns the model expects (from error analysis)
            for col in CATEGORICAL_COLS:
                if col == 'MeasureItem':
                    future_exog_df[col] = 'Monthly Hourly Load Values'
                elif col == 'CountryCode':
                    future_exog_df[col] = 'NL'
                elif col == 'Time_of_Day':
                    # Infer Time_of_Day (Day: 6-17, Night: 18-5)
                    future_exog_df[col] = np.where(future_exog_df.index.hour.isin(range(6, 18)), 'Day', 'Night')
                elif col == 'Detailed_Time_of_Day':
                    # Infer Detailed_Time_of_Day based on hour to match model features
                    future_exog_df[col] = future_exog_df.index.hour.map(lambda h: 'Morning' if h in range(5, 12) else 'Noon' if h in range(12, 17) else 'Evening' if h in range(17, 22) else 'Midnight')
                    # NOTE: Using 'Midnight' for 22, 23, 0, 1, 2, 3, 4. This is the simplest mapping that generates the required dummies from the error list.
                elif col in ['CreateDate', 'UpdateDate']:
                    # Use the most frequent date string that generated the required dummy column
                    future_exog_df[col] = '03-03-2025 12:24:13' 

            # Perform the encoding on the future data
            future_exog_df = pd.get_dummies(future_exog_df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
            
            # -----------------------------------------------------------------------------------------
            
            # Combine historical (actual demand) and future exogenous data
            df_combined = pd.concat([historical_df[[TARGET_COL_SANITIZED]], future_exog_df], axis=0)
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

            # Select only the needed future index
            idx_forecast = pd.date_range(start=forecast_start_dt, end=forecast_end_dt, freq='h')
            future_df = df_combined.loc[idx_forecast].copy()

            # Ensure future temperature data is correctly included for feature creation
            future_df = future_df.loc[:, ~future_df.columns.duplicated()].copy()
            future_df = create_features(future_df.copy())
            
            # --- CRITICAL FIX: Initialize all Missing Numerical/Exogenous and Lag Columns ---

            # Initialize Target Lag Columns (set to NaN, will be populated recursively)
            future_df[f'{TARGET_COL_SANITIZED}_lag24'] = np.nan
            future_df[f'{TARGET_COL_SANITIZED}_lag48'] = np.nan
            future_df[f'{TARGET_COL_SANITIZED}_roll72'] = np.nan
            
            # Initialize Other Missing Numerical/Exogenous Columns (from the error list)
            # These were generated in the training data but might be missing in the 'raw' future slice
            # We initialize them to NaN/0 and rely on create_features/get_dummies to populate the actual values
            MISSING_NUM_COLS = ['index', 'Cov_ratio', 'Number_of_Stations', 'Mean_Wind_Speed_0_1_m_s', 
                                'Maximum_Wind_Gust_0_1_m_s', 'Sunshine_Duration_0_1_hours', 
                                'Precipitation_Duration_0_1_hours', 'Hourly_Precipitation_Amount_0_1_mm', 
                                'Air_Pressure_0_1_hPa', 'Horizontal_Visibility', 'Cloud_Cover_octants', 
                                'Relative_Atmospheric_Humidity', 'Present_Weather_Code', 
                                'Present_Weather_Code_Indicator', 'Fog_0_no_1_yes', 'Rainfall_0_no_1_yes', 
                                'Snow_0_no_1_yes', 'Thunder_0_no_1_yes', 'Ice_Formation_0_no_1_yes', 
                                'Global_Radiation_kW_m2']
            
            for col in MISSING_NUM_COLS:
                if col not in future_df.columns:
                    # Initialize missing columns, assuming they should be 0 or NaN if not directly calculated
                    future_df[col] = 0.0

            # -----------------------------------------------------------------------------------

            # --- Recursive Prediction Loop (Simplified for Streamlit) ---

            PREDICTION_COL_NAME = 'Predicted_Demand_MW'
            TEMP_COL_SANITIZED = sanitize_feature_names([TEMP_COL])[0]

            # Need last 72 hours of actual data for initial roll72 calculation
            last_actuals = historical_df[TARGET_COL_SANITIZED].tail(72).copy()

            # Initialize prediction series (will be populated during loop)
            s_predictions = pd.Series(index=idx_forecast, dtype=float)

            # Get the feature names the model expects
            EXPECTED_FEATURES = model.feature_name_

            st.write(f"Forecasting from **{forecast_start_dt}** to **{forecast_end_dt}** (Total: {len(idx_forecast)} hours)")


            for i, current_index in enumerate(idx_forecast):

                # 1. Prepare the row for the current timestamp
                # Note: future_df already has all features EXCEPT the target lags
                X_current = future_df.loc[[current_index]].copy()

                # 2. Combine all known/predicted values for lag calculation
                s_latest = pd.concat([last_actuals, s_predictions.dropna()]).sort_index()

                # --- ROBUST LAG CALCULATION ---
                # Lag 24 & Lag 48
                X_current.loc[current_index, f'{TARGET_COL_SANITIZED}_lag24'] = s_latest.get(current_index - pd.Timedelta(hours=24))
                X_current.loc[current_index, f'{TARGET_COL_SANITIZED}_lag48'] = s_latest.get(current_index - pd.Timedelta(hours=48))

                # Rolling Mean 72
                roll72_window_end = current_index - pd.Timedelta(hours=24)
                roll72_window_start = roll72_window_end - pd.Timedelta(hours=72)
                roll72_val = s_latest.loc[roll72_window_start:roll72_window_end].mean()
                X_current.loc[current_index, f'{TARGET_COL_SANITIZED}_roll72'] = roll72_val
                # ------------------------------

                # 3. Predict the current time step
                X_current_features = X_current[EXPECTED_FEATURES].copy()

                # CRITICAL CHECK: Only predict after the first 72 hours (when target lags are available)
                if i >= 72:
                    prediction = model.predict(X_current_features)[0]
                else:
                    prediction = np.nan

                # 4. Assign the prediction back into the series for the next step
                s_predictions.loc[current_index] = prediction


            # --- Display Results ---
            future_df[PREDICTION_COL_NAME] = s_predictions
            # Drop the first 72 hours of NaN predictions and any rows where prediction failed
            df_plot = future_df.dropna(subset=[PREDICTION_COL_NAME]) 

            st.success("Forecast Complete! Results displayed below.")

            # Shortage analysis (Simplified)
            # Define a simple shortage threshold (e.g., top 1% of historical demand)
            shortage_threshold = last_actuals.quantile(0.99)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Peak Predicted Demand", value=f"{df_plot[PREDICTION_COL_NAME].max():,.2f} MW")
            with col2:
                st.metric(label="Shortage Threshold (99th Percentile)", value=f"{shortage_threshold:,.2f} MW")

            shortage_hours = df_plot[df_plot[PREDICTION_COL_NAME] > shortage_threshold]

            if not shortage_hours.empty:
                st.warning(f"🚨 **SHORTAGE ALERT:** Predicted demand exceeds the 99th percentile threshold during **{len(shortage_hours)} hours** in the forecast period.")
                st.dataframe(shortage_hours.sort_values(PREDICTION_COL_NAME, ascending=False).head(), use_container_width=True)
            else:
                st.info("No extreme shortage events predicted above the 99th percentile threshold.")


            # Plotting the forecast
            st.subheader("Hourly Demand Forecast")
            st.line_chart(df_plot, y=PREDICTION_COL_NAME)

            st.subheader("Raw Data Preview")
            st.dataframe(df_plot[[PREDICTION_COL_NAME] + [f for f in EXPECTED_FEATURES if f in df_plot.columns]].head(10), use_container_width=True)
