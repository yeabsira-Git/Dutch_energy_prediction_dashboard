import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta, time # Import time for easier date manipulation
import lightgbm as lgb
import altair as alt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
MODEL_FILENAME = 'lightgbm_demand_model.joblib'

# List of categorical columns from the original dataset
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']
TARGET_COL_SANITIZED = 'Demand_MW' 

# Define prediction limits
FORECAST_START_DATE_LIMIT = datetime(2025, 7, 1)
FORECAST_END_DATE_LIMIT = datetime(2025, 12, 31, 23, 0, 0) # Inclusive end time

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

def create_features(df):
    """
    Creates all time-based, lag, and rolling window features expected by the model.
    This function must be consistent with the features generated during training.
    """
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    # 1. Time Features (Essential for both historical and forecast data)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    # Reconstructed feature from the error message
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int) 
    
    # 2. Demand Lag and Rolling Features
    if TARGET_COL_SANITIZED in df.columns:
        df['Demand_MW_lag24'] = df[TARGET_COL_SANITIZED].shift(24)
        df['Demand_MW_lag48'] = df[TARGET_COL_SANITIZED].shift(48)
        df['Demand_MW_roll72'] = df[TARGET_COL_SANITIZED].shift(24).rolling(window=72).mean()

    # 3. Temperature Lag and Rolling Features (Reconstructed from error list)
    if TEMP_COL_SAN in df.columns:
        df['temp_lag24'] = df[TEMP_COL_SAN].shift(24)
        df['temp_roll72'] = df[TEMP_COL_SAN].shift(24).rolling(window=72).mean()
        df['temp_roll168'] = df[TEMP_COL_SAN].shift(24).rolling(window=168).mean()
        
    # Return all generated columns (minus the target itself)
    feature_cols = [col for col in df.columns if col != TARGET_COL_SANITIZED]
    return df[feature_cols]

# --- 2. CACHING AND LOADING ---

@st.cache_data(show_spinner="Loading and aligning historical data...")
def load_data(file_path):
    """
    Loads, sanitizes all columns, applies one-hot encoding, and prepares historical data.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=[DATE_COL], index_col=DATE_COL)
        df.columns = sanitize_feature_names(df.columns)
        df = df.resample('H').first()
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dummy_na=False)
        df = df.dropna(subset=[TARGET_COL_SANITIZED])
        df = df.iloc[168:] 
        return df
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

# --- 3. RECURSIVE FORECASTING CORE LOGIC ---

def _run_recursive_forecast_core(historical_df, model, forecast_steps):
    """
    Core function that runs a step-by-step recursive forecast.
    It predicts the full path required, including any gap between
    the historical end and the desired forecast period.
    """
    
    last_known_time = historical_df.index[-1]
    forecast_index = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                   periods=forecast_steps, 
                                   freq='H')

    # Get the static/weather columns from the historical data
    static_weather_cols = [col for col in historical_df.columns if col != TARGET_COL_SANITIZED]
    
    # 1. Setup future DataFrame by tiling recent historical weather data (168h cycle)
    df_forecast = pd.DataFrame(index=forecast_index)
    
    for col in static_weather_cols:
        if col in historical_df.columns:
            historical_slice = historical_df[col].iloc[-168:]
            tiled_data = np.tile(historical_slice.values, (forecast_steps // 168) + 1)[:forecast_steps]
            df_forecast[col] = tiled_data
        else:
            df_forecast[col] = 0 
            
    # CRITICAL: Ensure OHE features for the future index are correctly set based on the date/time
    temp_future_df = pd.DataFrame(index=forecast_index)
    temp_future_df['Time_of_Day'] = temp_future_df.index.hour.map(lambda h: 'Day' if 6 <= h < 18 else 'Night')
    temp_future_df['Detailed_Time_of_Day'] = temp_future_df.index.hour.map({
        0: 'Midnight', 1: 'Midnight', 2: 'Midnight', 3: 'Midnight', 4: 'Midnight', 5: 'Midnight',
        6: 'Morning', 7: 'Morning', 8: 'Morning', 9: 'Morning', 10: 'Morning', 11: 'Morning',
        12: 'Noon', 13: 'Noon', 14: 'Noon', 15: 'Noon', 16: 'Noon', 17: 'Noon',
        18: 'Evening', 19: 'Evening', 20: 'Evening', 21: 'Evening', 22: 'Evening', 23: 'Evening'
    }) 

    future_ohe_df = pd.get_dummies(temp_future_df[['Time_of_Day', 'Detailed_Time_of_Day']], dummy_na=False)

    for col in future_ohe_df.columns:
        ohe_col_name = sanitize_feature_names([col])[0]
        if ohe_col_name in df_forecast.columns:
            df_forecast[ohe_col_name] = future_ohe_df[col].values
            
    df_forecast[TARGET_COL_SANITIZED] = np.nan # This will hold our predictions

    # Combine historical and future data
    df_combined = pd.concat([historical_df, df_forecast])
    
    # 2. Perform Recursive Loop
    for t in forecast_index:
        df_temp = df_combined.loc[:t].copy() 
        features_t_raw = create_features(df_temp.tail(168)).tail(1)
        
        try:
            X_t = features_t_raw.reindex(columns=model.feature_name_, fill_value=0)
        except Exception as e:
            st.error(f"Failed to align features for prediction: {e}")
            return pd.DataFrame()

        pred_t = model.predict(X_t)[0]
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
    
    last_known_time = historical_df.index[-1]
    
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
    st.subheader("2.1. Interactive Historical Daily Pattern Viewer")
    st.markdown("Select a historical date to inspect the 24-hour energy demand (MW) versus temperature (°C) for that specific day, highlighting the typical evening peak.")
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    dates = historical_df.index.normalize().unique()
    default_date = dates[-1].to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Historical Date",
        value=default_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date(),
        key='historical_date_picker'
    )

    selected_day_df = historical_df[historical_df.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No data available for {selected_date}.")
        return

    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df[TARGET_COL_SANITIZED]
    
    if TEMP_COL_SAN in selected_day_df.columns:
        selected_day_df['Temperature_C'] = selected_day_df[TEMP_COL_SAN] / 10.0
    else:
        selected_day_df['Temperature_C'] = 0 
    
    # Altair Charts for dual axis plot
    base = alt.Chart(selected_day_df).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day')),
    )

    demand_chart = base.mark_line(point=True, color='#006494').encode(
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Demand (MW)', titleColor='#006494')),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f')]
    ).properties(
        title=f"Demand and Temperature on {selected_date.strftime('%Y-%m-%d')}"
    )

    temp_chart = base.mark_line(point=True, color='#E9573E').encode(
        y=alt.Y('Temperature_C:Q', axis=alt.Axis(title='Temperature (°C)', titleColor='#E9573E')),
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
    st.title("💡 Predictive Energy Demand Risk Platform")
    st.markdown("#### Daily forecast for Electricity Demand peaks of The Netherlands")

    # Load resources
    historical_df = load_data('cleaned_energy_weather_data(1).csv')
    model = load_model(MODEL_FILENAME)
    
    if historical_df.empty or model is None:
        return

    # 1. Sidebar Info
    st.sidebar.header("Data & Model Info")
    st.sidebar.success(f"Historical Data Loaded: {historical_df.shape[0]} records")
    st.sidebar.success(f"Model Loaded: {model.__class__.__name__}")
    st.sidebar.info(f"Last Actual Demand Date: {historical_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    # 2. Display Historical EDA
    #st.subheader("2. Exploratory Data Analysis (EDA)")
    display_historical_daily_pattern(historical_df)

    # 3. Daily Forecast Controls and Execution
    st.subheader("3. Single-Day Peak Forecast")
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
            
            df_forecast = run_daily_forecast(historical_df, model, target_date)
            
            if not df_forecast.empty:
                st.session_state.daily_forecast = df_forecast
                
                # Instant Peak Alert (to replace JS alert())
                peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
                peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
                peak_time_interval = f"{peak_row_index.hour:02d}:00 - {peak_row_index.hour+1:02d}:00"
                peak_category = map_hour_to_detailed_time_of_day(peak_row_index.hour)
                
                if 'Evening' in peak_category:
                    st.toast(f"🚨 PEAK ALERT! Predicted peak of {peak_demand:,.0f} MW at {peak_time_interval} (Evening) on {target_date.strftime('%Y-%m-%d')}!", icon='🔥')
                else:
                    st.toast(f"✅ Prediction complete. Peak of {peak_demand:,.0f} MW at {peak_time_interval} on {target_date.strftime('%Y-%m-%d')}.", icon='💡')

    # 4. Display Daily Forecast Results
    st.subheader("4. Forecast Results")
    
    if 'daily_forecast' in st.session_state and not st.session_state.daily_forecast.empty:
        
        # Display the 24-hour chart
        display_daily_forecast_chart(st.session_state.daily_forecast, target_date)
        
        # Display the peak summary and risk warning
        display_daily_peak_summary(st.session_state.daily_forecast, target_date)
        
    else:
        st.info("Click 'Run Daily Forecast' above to generate the prediction for a single day.")
    
if __name__ == '__main__':
    main()
