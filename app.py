import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta, time
import lightgbm as lgb
import altair as alt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- 1. UTILITY FUNCTIONS (Feature Engineering & Sanitization) ---

def sanitize_feature_names(columns):
    """Helper function to sanitize column names for consistency."""
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

# --- CONFIGURATION (Must match training script) ---
# NOTE: This block is moved here to ensure sanitize_feature_names is defined first (FIX for NameError).
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
TEMP_COL_SANITIZED = sanitize_feature_names([TEMP_COL])[0] 
TARGET_COL_SANITIZED = 'Demand_MW' 

# List of categorical columns from the original dataset
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# Define prediction limits
FORECAST_START_DATE_LIMIT = datetime(2025, 7, 1)
FORECAST_END_DATE_LIMIT = datetime(2025, 12, 31, 23, 0, 0) # Inclusive end time


def create_features(df):
    """
    Creates all time-based, lag, and rolling window features expected by the model.
    """
    
    # 1. Time Features (Essential for both historical and forecast data)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int) 
    
    # 2. Demand Lag and Rolling Features (Used for historical calculation only)
    if TARGET_COL_SANITIZED in df.columns:
        df['Demand_MW_lag24'] = df[TARGET_COL_SANITIZED].shift(24)
        df['Demand_MW_lag48'] = df[TARGET_COL_SANITIZED].shift(48)
        df['Demand_MW_roll72'] = df[TARGET_COL_SANITIZED].shift(24).rolling(window=72).mean()

    # 3. Temperature Lag and Rolling Features
    if TEMP_COL_SANITIZED in df.columns:
        df['temp_lag24'] = df[TEMP_COL_SANITIZED].shift(24)
        # Shift(1) is removed here for rolling means calculated on the combined set,
        # as the 'create_features' runs on a copy of the combined set *before* the loop.
        df['temp_roll72'] = df[TEMP_COL_SANITIZED].shift(24).rolling(window=72, min_periods=1).mean()
        df['temp_roll168'] = df[TEMP_COL_SANITIZED].shift(24).rolling(window=168, min_periods=1).mean()
        
    # Return all generated columns (minus the target itself)
    return df

# --- 2. CACHING AND LOADING (MODIFIED TO INCLUDE CLIMATOLOGY) ---

@st.cache_data(show_spinner="Loading and aligning historical data...")
def load_data(file_path):
    """
    Loads, sanitizes all columns, and PREPARES THE 2025 FORECAST FRAME
    by appending 2024 climatology for temperature (FIX for unrealistic weather input).
    """
    try:
        df_hist = pd.read_csv(file_path, parse_dates=[DATE_COL], index_col=DATE_COL)
        df_hist.columns = sanitize_feature_names(df_hist.columns)
        df_hist = df_hist.resample('H').first()
        
        # --- CRITICAL FIX: Append 2024 Climatology for Exogenous Variables ---
        
        DATASET_END_ACTUAL = df_hist.index[-1]
        
        START_2024_CLIMATOLOGY = datetime(2024, 7, 1, 0, 0)
        END_2024_CLIMATOLOGY = datetime(2024, 12, 31, 23, 0)

        # 1. Grab 2024 temperature history
        temp_history_climatology = df_hist.loc[START_2024_CLIMATOLOGY:END_2024_CLIMATOLOGY, TEMP_COL_SANITIZED].copy()
        
        # 2. Define 2025 forecast dates
        dates_2025_forecast = pd.date_range(start=DATASET_END_ACTUAL + timedelta(hours=1), 
                                            end=FORECAST_END_DATE_LIMIT, 
                                            freq='H')
        
        # Handle cases where the data doesn't align perfectly 
        if len(temp_history_climatology) < len(dates_2025_forecast):
             # Pad with last known value if 2024 period is shorter than 2025
             pad_needed = len(dates_2025_forecast) - len(temp_history_climatology)
             temp_history_climatology = pd.concat([temp_history_climatology, 
                                                   pd.Series([temp_history_climatology.iloc[-1]] * pad_needed)])
        
        # 3. Create the 2025 temperature forecast series
        temp_forecast = pd.Series(temp_history_climatology.values[:len(dates_2025_forecast)], 
                                  index=dates_2025_forecast)
        temp_forecast.name = TEMP_COL_SANITIZED

        # 4. Create the future data frame and fill temperature
        future_exog_df = pd.DataFrame(index=dates_2025_forecast)
        future_exog_df[TEMP_COL_SANITIZED] = temp_forecast
        future_exog_df[TARGET_COL_SANITIZED] = np.nan # Target is NaN for forecast
        
        # 5. Concatenate all data (Historical + Future with 2024 Temp)
        df_full = pd.concat([df_hist, future_exog_df])
        
        # --- Standard Processing Steps ---
        
        # Fill other non-temp/non-target columns (i.e., OHE columns) with 0.0 in the future period
        # OHE columns are not yet generated, so this fillna must happen after OHE or we fill the raw categorical columns
        
        # Apply OHE to the full dataset to ensure alignment
        df_full = pd.get_dummies(df_full, columns=CATEGORICAL_COLS, dummy_na=False)

        # Now fill the OHE columns (which are now numeric) in the future frame with 0.0
        for col in df_full.columns:
            if col not in [TARGET_COL_SANITIZED, TEMP_COL_SANITIZED] and df_full[col].isna().any():
                df_full[col] = df_full[col].fillna(0.0)

        # Drop NaNs based on the historical target, but keep future NaNs
        df_historical_clean = df_full.loc[df_full.index <= DATASET_END_ACTUAL].dropna(subset=[TARGET_COL_SANITIZED])
        df_future = df_full.loc[df_full.index > DATASET_END_ACTUAL]
        
        # Recombine
        df_clean = pd.concat([df_historical_clean, df_future])
        
        # Ensure we have enough data for the initial lags (e.g., 168 hours)
        df_clean = df_clean.iloc[168:] 
        
        st.sidebar.info("✅ Future Temp uses **2024 Climatology**.")
        return df_clean
        
    except Exception as e:
        st.error(f"Error loading or processing data. Ensure 'cleaned_energy_weather_data(1).csv' is correctly formatted: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(file_path):
    """Loads the pre-trained LightGBM model."""
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. RECURSIVE FORECASTING CORE LOGIC (FIXED) ---

def _run_recursive_forecast_core(historical_df, model, forecast_steps):
    """
    Core function that runs a step-by-step recursive forecast.
    FIX: Removed the old "last 168 hours tiling" of weather features.
    """
    
    last_known_time = historical_df.index[historical_df[TARGET_COL_SANITIZED].notna()].max()
    forecast_index = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                     periods=forecast_steps, 
                                     freq='H')

    # The 'historical_df' passed in now contains the full historical data PLUS 
    # the 2025 temperature data (from 2024 climatology) appended at the end.
    df_combined = historical_df.copy()
    
    # Pre-calculate ALL time and weather features once on the combined frame
    # This step generates temp_lag24, temp_roll72/168 using the 2024 climatology data.
    df_combined = create_features(df_combined.copy())

    # Setup feature columns
    model_feature_names = model.feature_name_
    
    # 2. Perform Recursive Loop
    for t in forecast_index:
        
        # Get the feature row containing all pre-calculated time/weather features
        X_t = df_combined.loc[t].to_frame().T.copy()
        
        # --- OPTIMIZED FEATURE GENERATION (Only for DEMAND LAGS) ---
        
        # Calculate lag features using already set values in df_combined
        lag24_dt = t - timedelta(hours=24)
        lag48_dt = t - timedelta(hours=48)
        
        # Lag 24 and 48: Directly retrieve from df_combined (Actual or Predicted Demand)
        X_t['Demand_MW_lag24'] = df_combined.loc[lag24_dt, TARGET_COL_SANITIZED]
        X_t['Demand_MW_lag48'] = df_combined.loc[lag48_dt, TARGET_COL_SANITIZED]
        
        # Roll 72: Needs 72 hours of data *ending* 24 hours ago
        roll72_window_end = t - timedelta(hours=24)
        roll72_window_start = roll72_window_end - timedelta(hours=71) # 72 periods
        roll72_index = pd.date_range(start=roll72_window_start, end=roll72_window_end, freq='H')
        X_t['Demand_MW_roll72'] = df_combined.loc[roll72_index, TARGET_COL_SANITIZED].mean()
        
        # --- End Optimized Feature Generation ---

        try:
            # Align features with the model's expectation
            X_t = X_t.reindex(columns=model_feature_names, fill_value=0)
            
            # Use .astype(np.float64).values to ensure a pure numpy array with float dtype for prediction
            X_t_numeric = X_t.astype(np.float64).values
            
        except Exception as e:
            st.error(f"Failed to align features for prediction: {e}")
            return pd.DataFrame()

        # Predict and update the combined DataFrame for the next iteration
        pred_t = model.predict(X_t_numeric)[0]
        df_combined.loc[t, TARGET_COL_SANITIZED] = pred_t

    # Final forecast is the predicted portion of the combined DataFrame
    final_forecast_df = df_combined.loc[forecast_index].rename(
        columns={TARGET_COL_SANITIZED: 'Predicted_Demand_MW'}
    )
    
    return final_forecast_df

# --- 4. DAILY FORECAST EXECUTION ---

def run_daily_forecast(historical_df, model, target_date):
    """
    Performs a recursive forecast from the last historical point up to the end of the target_date,
    and returns only the 24 hours of the target_date.
    """
    
    # Find the last time step with actual demand data
    last_known_time = historical_df[historical_df[TARGET_COL_SANITIZED].notna()].index[-1]
    
    # Target period: 00:00 on target_date to 23:00 on target_date (24 hours)
    start_time_of_day = datetime.combine(target_date, time(0, 0))
    end_time_of_day = datetime.combine(target_date, time(23, 0))
    
    if start_time_of_day <= last_known_time:
        st.warning(f"The date {target_date.strftime('%Y-%m-%d')} is in the historical data. Showing actual data.")
        return historical_df[historical_df.index.date == target_date].rename(
            columns={TARGET_COL_SANITIZED: 'Predicted_Demand_MW'}
        )

    # Calculate the number of hours to predict: from the hour *after* last_known_time up to and including end_time_of_day
    hours_to_predict = int((end_time_of_day - last_known_time).total_seconds() / 3600)

    if hours_to_predict <= 0:
           st.error(f"The selected date is too close to the last known data point ({last_known_time.strftime('%Y-%m-%d %H:%M')}). Please select a later date.")
           return pd.DataFrame()
           
    with st.spinner(f"Running recursive forecast for {hours_to_predict} hours up to {target_date.strftime('%Y-%m-%d')}..."):
        # Run the core recursive logic to predict the entire path (gap + target day)
        df_full_path = _run_recursive_forecast_core(historical_df, model, hours_to_predict)

    # Filter the result to just the 24 hours of the target_date
    daily_forecast = df_full_path.loc[start_time_of_day:end_time_of_day]
    
    return daily_forecast


# --- 5. STREAMLIT APP LAYOUT FUNCTIONS ---

def map_hour_to_detailed_time_of_day(hour):
    """Maps an hour (0-23) to a readable time segment."""
    if 0 <= hour < 6: return 'Night (00:00-05:59)'
    if 6 <= hour < 12: return 'Morning (06:00-11:59)'
    if 12 <= hour < 18: return 'Noon/Afternoon (12:00-17:59)'
    if 18 <= hour < 24: return 'Evening (18:00-23:59)'
    return 'Unknown'

def display_historical_daily_pattern(historical_df):
    """Allows user to select a date and views its 24-hour demand and temperature pattern."""
    st.subheader("1. Interactive Historical Daily Pattern Viewer")
    st.markdown("Select a historical date to inspect the 24-hour energy demand (MW) versus temperature (°C) for that specific day, highlighting the typical peak.")
    
    dates = historical_df.index.normalize().unique()
    default_date = dates[dates <= datetime(2025, 6, 30)].max().to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Historical Date",
        value=default_date,
        min_value=historical_df.index.min().to_pydatetime().date(),
        max_value=default_date, # Limit to the last actual historical day
        key='historical_date_picker'
    )

    selected_day_df = historical_df[historical_df.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No data available for {selected_date}.")
        return

    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df[TARGET_COL_SANITIZED]
    
    if TEMP_COL_SANITIZED in selected_day_df.columns:
        selected_day_df['Temperature_C'] = selected_day_df[TEMP_COL_SANITIZED] / 10.0
    else:
        selected_day_df['Temperature_C'] = 0 
    
    # Altair Charts for dual axis plot
    base = alt.Chart(selected_day_df).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day')),
    )

    demand_chart = base.mark_line(point=True, color="#940011").encode(
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Demand (MW)', titleColor="#940000")),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f')]
    ).properties(
        title=f"Demand and Temperature on {selected_date.strftime('%Y-%m-%d')}"
    )

    temp_chart = base.mark_line(point=True, color="#3E66E9").encode(
        y=alt.Y('Temperature_C:Q', axis=alt.Axis(title='Temperature (°C)', titleColor="#3E5DE9")),
        tooltip=['Hour:O', alt.Tooltip('Temperature_C:Q', format='.1f')]
    )
    
    final_chart = alt.layer(demand_chart, temp_chart).resolve_scale(
        y='independent'
    ).interactive()
    
    st.altair_chart(final_chart, use_container_width=True)
    st.caption("Energy demand (blue) typically ramps up sharply in the evening, often inversely correlated with temperature (orange).")
    st.markdown("---")


def display_daily_peak_summary(selected_day_df, selected_date, prediction_mode=True):
    """
    Analyzes the selected day's data (actual or forecast) and extracts the peak demand time and category.
    """
    
    peak_column = 'Predicted_Demand_MW' if prediction_mode else 'Demand_MW'
    
    if selected_day_df.empty or peak_column not in selected_day_df.columns:
        return

    st.subheader(f"4.1. Peak Demand Analysis for {selected_date.strftime('%Y-%m-%d')}")
    
    peak_demand_row = selected_day_df[peak_column].idxmax()
    peak_demand = selected_day_df.loc[peak_demand_row, peak_column]
    peak_hour = peak_demand_row.hour
    
    peak_category = map_hour_to_detailed_time_of_day(peak_hour)
    peak_time_interval = f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00"
    
    is_evening_peak = 'Evening' in peak_category
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Peak Demand (MW)", f"{peak_demand:,.0f}")
    
    with col2:
        st.metric("Peak Time Interval", peak_time_interval)
        
    with col3:
        # Custom display for the Evening Peak status
        status_color = '#155724' if is_evening_peak else '#0c5460'
        bg_color = '#d4edda' if is_evening_peak else '#d1ecf1'
        border_color = '#c3e6cb' if is_evening_peak else '#bee5eb'
        
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; text-align: center; 
                    background-color: {bg_color};
                    border: 1px solid {border_color};">
            <p style="margin: 0; font-size: 14px; font-weight: 600;">Peak Time Category</p>
            <h4 style="margin: 5px 0 0; color: {status_color};">{peak_category.split('(')[0].strip()}</h4>
        </div>
        """, unsafe_allow_html=True)
        
    if is_evening_peak:
        st.success("**HIGH RISK:** The predicted daily peak is driven by the high-risk **Evening** consumption window. Action is required for this time slot.")
    else:
        st.info("The predicted daily peak occurred outside the typical Evening high-risk window, which is often less critical but still requires monitoring.")
    st.markdown("---")

def display_daily_forecast_chart(selected_day_df, selected_date):
    """Displays the 24-hour chart for the forecast day."""
    
    st.subheader("4. Predicted 24-Hour Demand Pattern")
    
    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df['Predicted_Demand_MW']
    
    # Add the categorical time of day column
    selected_day_df['Time_of_Day_Category'] = selected_day_df['Hour'].apply(map_hour_to_detailed_time_of_day)

    # Define order for categorical variable for clean legend/coloring
    category_order = ['Night (00:00-05:59)', 'Morning (06:00-11:59)', 'Noon/Afternoon (12:00-17:59)', 'Evening (18:00-23:59)']
    
    chart = alt.Chart(selected_day_df).mark_line(point=True).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day (0-23)')),
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Predicted Demand (MW)')),
        color=alt.Color('Time_of_Day_Category:N', sort=category_order, title="Time of Day"),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f'), 'Time_of_Day_Category:N']
    ).properties(
        title=f"Predicted Hourly Demand by Time Segment on {selected_date.strftime('%Y-%m-%d')}"
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.caption("The colored segments highlight how collective human behavior drives demand peaks.")
    
# --- MAIN EXECUTION ---

def main():
    st.set_page_config(layout="wide")
    st.title("💡 Predictive Dutch Energy Demand Platform: Peak Alert System")

    # Load resources
    # Load data now returns the full historical + 2025 forecast frame with 2024 temp
    historical_df = load_data('cleaned_energy_weather_data(1).csv') 
    model = load_model(MODEL_FILENAME)
    
    if historical_df.empty or model is None:
        return

    # 1. Sidebar Info
    st.sidebar.header("Data & Model Info")
    st.sidebar.success(f"Historical Data Loaded: {historical_df.shape[0]} records")
    st.sidebar.success(f"Model Loaded: {model.__class__.__name__}")
    
    # Find the true last actual demand date (before the NaNs start)
    last_actual_date = historical_df[historical_df[TARGET_COL_SANITIZED].notna()].index[-1]
    st.sidebar.info(f"Last Actual Demand Date: {last_actual_date.strftime('%Y-%m-%d %H:%M')}")
    
    # 2. Display Historical EDA
    display_historical_daily_pattern(historical_df)

    # 3. Daily Forecast Controls and Execution
    st.subheader("2. Single-Day Peak Forecast")
    st.info(f"Select a day between **{FORECAST_START_DATE_LIMIT.strftime('%Y-%m-%d')}** and **{FORECAST_END_DATE_LIMIT.strftime('%Y-%m-%d')}** to run a minimal recursive prediction.")

    col_date, col_btn = st.columns([0.7, 0.3])
    
    with col_date:
        target_date = st.date_input(
            "Target Date to Predict (2025)",
            value=FORECAST_START_DATE_LIMIT.date(),
            min_value=FORECAST_START_DATE_LIMIT.date(),
            max_value=FORECAST_END_DATE_LIMIT.date(),
            key='target_date_picker'
        )
    
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
        if st.button(f'⚡ Run Daily Forecast', key='run_daily_forecast_btn'):
            # Clear previous result
            if 'daily_forecast' in st.session_state:
                del st.session_state.daily_forecast
            
            # Pass the full df_clean (historical + future temp) to run_daily_forecast
            df_forecast = run_daily_forecast(historical_df, model, target_date)
            
            if not df_forecast.empty:
                st.session_state.daily_forecast = df_forecast
                
                # Instant Peak Alert
                peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
                peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
                peak_time_interval = f"{peak_row_index.hour:02d}:00 - {peak_row_index.hour+1:02d}:00"
                peak_category = map_hour_to_detailed_time_of_day(peak_row_index.hour)
                
                if 'Evening' in peak_category:
                    st.toast(f"🚨 PEAK ALERT! Predicted peak of {peak_demand:,.0f} MW at {peak_time_interval} (Evening) on {target_date.strftime('%Y-%m-%d')}!", icon='🔥')
                else:
                    st.toast(f"✅ Prediction complete. Peak of {peak_demand:,.0f} MW at {peak_time_interval} on {target_date.strftime('%Y-%m-%d')}.", icon='💡')

    # 4. Display Daily Forecast Results
    st.subheader("3. Forecast Results")
    
    if 'daily_forecast' in st.session_state and not st.session_state.daily_forecast.empty:
        
        # Display the 24-hour chart
        display_daily_forecast_chart(st.session_state.daily_forecast, target_date)
        
        # Display the peak summary and risk warning
        display_daily_peak_summary(st.session_state.daily_forecast, target_date)
        
    else:
        st.info("Click 'Run Daily Forecast' above to generate the prediction for a single day.")
    
if __name__ == '__main__':
    main()