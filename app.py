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

    # Sanitization for consistent column names
    df.columns = sanitize_feature_names(df.columns)

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

@st.cache_data
def load_data(path):
    """Load and preprocess the historical data."""
    try:
        df = pd.read_csv(path)
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
            # FIX 2: Corrected function call by removing 'path_only=True' and space in filename
            df_full_temp = load_data('cleaned_energy_weather_data(1).csv')
            df_full_temp.set_index(DATE_COL, inplace=True)
            df_full_temp.index = pd.to_datetime(df_full_temp.index)

            # --- Recreate 2024 Climatology Forecast for Temperature (Exogenous Variable) ---
            START_2024_CLIMATOLOGY = '2024-07-01 00:00:00'
            END_2024_CLIMATOLOGY = '2024-12-31 23:00:00'

            # Note: This block assumes 'df_full_temp' contains the 'Temperature (0.1 degrees Celsius)' column, 
            # which is true based on your script's logic.

            temp_history_climatology = df_full_temp.loc[START_2024_CLIMATOLOGY:END_2024_CLIMATOLOGY, TEMP_COL].copy()

            # Create the future index that matches the forecast period
            dates_2025_forecast_index = pd.date_range(start=datetime(2025, 7, 1), end=datetime(2025, 12, 31, 23, 0, 0), freq='h')

            if len(temp_history_climatology) != len(dates_2025_forecast_index):
                st.error("Climatology data length mismatch. Cannot proceed.")
                st.stop()

            temp_forecast = pd.Series(temp_history_climatology.values, index=dates_2025_forecast_index)
            temp_forecast.name = TEMP_COL

            future_exog_df = temp_forecast.to_frame()
            future_exog_df[TARGET_COL] = np.nan

            # Combine historical and future exog data
            df_combined = pd.concat([historical_df[[sanitize_feature_names([TARGET_COL])[0]]], future_exog_df], axis=0)
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')]

            # Select only the needed future index
            idx_forecast = pd.date_range(start=forecast_start_dt, end=forecast_end_dt, freq='h')
            future_df = df_combined.loc[idx_forecast].copy()

            # Ensure future temperature data is correctly included for feature creation
            future_df = future_df.loc[:, ~future_df.columns.duplicated()].copy()
            future_df = create_features(future_df.copy())

            # --- Recursive Prediction Loop (Simplified for Streamlit) ---

            PREDICTION_COL_NAME = 'Predicted_Demand_MW'
            TARGET_COL_SANITIZED = sanitize_feature_names([TARGET_COL])[0]
            TEMP_COL_SANITIZED = sanitize_feature_names([TEMP_COL])[0]


            # Need last 72 hours for initial roll72 calculation
            last_actuals = historical_df[TARGET_COL_SANITIZED].tail(72).copy()

            # Initialize prediction series (will be populated during loop)
            s_predictions = pd.Series(index=idx_forecast, dtype=float)

            # Get the feature names the model expects
            EXPECTED_FEATURES = model.feature_name_

            st.write(f"Forecasting from **{forecast_start_dt}** to **{forecast_end_dt}** (Total: {len(idx_forecast)} hours)")


            for i, current_index in enumerate(idx_forecast):

                # 1. Prepare the row for the current timestamp
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
            future_df = future_df.dropna(subset=[PREDICTION_COL_NAME]) # Drop the first 72 hours of NaN predictions

            st.success("Forecast Complete! Results displayed below.")

            # Create a combined data frame for plotting and display
            df_plot = future_df[[PREDICTION_COL_NAME]].copy()

            # Shortage analysis (Simplified)
            # Define a simple shortage threshold (e.g., top 1% of historical demand)
            shortage_threshold = last_actuals.quantile(0.99)

            st.metric(label="Peak Predicted Demand", value=f"{df_plot[PREDICTION_COL_NAME].max():.2f} MW")
            st.metric(label="Shortage Threshold (99th Percentile)", value=f"{shortage_threshold:.2f} MW")

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
            st.dataframe(df_plot, use_container_width=True)