import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, date
import lightgbm as lgb
import altair as alt 
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' 
CAPACITY_THRESHOLD = 15000 # Example hard limit (in MW). ADJUST THIS TO YOUR NETWORK'S CAPACITY!
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# --- SCENARIO MAPPING ---
SCENARIO_MAP = {
    "1. Cold (0°C - 10°C)": 5.0,     
    "2. Mild (10°C - 20°C)": 15.0,   
    "3. Warm (20°C - 25°C)": 22.5,   
    "4. Summer (> 25°C)": 30.0      
}

# --- TIME PERIOD MAPPING (RESTORED VERBOSE NAMES) ---
def map_hour_to_period(hour):
    """Maps the hour (0-23) to a Time of Day category."""
    if 0 <= hour <= 5:
        return 'Midnight (00:00 - 05:59)'
    elif 6 <= hour <= 11:
        return 'Morning (06:00 - 11:59)'
    elif 12 <= hour <= 16:
        return 'Noon (12:00 - 16:59)'
    elif 17 <= hour <= 23:
        return 'Evening (17:00 - 23:59)'
    return 'Other'


# --- 1. UTILITY FUNCTIONS ---

def sanitize_feature_names(columns):
    """Sanitizes feature names to be compatible with LightGBM and pandas."""
    new_cols = []
    for col in columns:
        col = str(col)
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        col = re.sub(r'^_+|_+$', '', col)
        col = re.sub(r'_{2,}', '_', col)
        new_cols.append(col)
    return new_cols

@st.cache_data
def create_features(df):
    """Creates basic time and temperature features."""
    
    if DATE_COL in df.columns:
        if DATE_COL not in df.index.names:
             df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True) 
             df = df.set_index(DATE_COL)
        
    # Basic Time Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Additional Simple Features 
    df['time_index'] = np.arange(len(df)) + 1 
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Temperature Conversion
    if TEMP_COL in df.columns:
        df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10
    
    # Simple lag 24 for historical data 
    if TARGET_COL in df.columns:
        df['lag_24'] = df[TARGET_COL].shift(24) 
    
    return df

# --- 2. DATA/MODEL LOADING ---

@st.cache_data
def load_data():
    """Loads and preprocesses HISTORICAL data (using your CSV)."""
    data = pd.read_csv('cleaned_energy_weather_data(1).csv')
    data = create_features(data)
    
    # Perform One-Hot Encoding on the historical data
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=False)
    
    # Sanitize all columns (including new OHE ones)
    data.columns = sanitize_feature_names(data.columns)
    
    # Forward fill (ffill) weather/lag data if any NaNs remain
    data = data.ffill()
    return data

@st.cache_resource
def load_model():
    """Loads the trained LightGBM model."""
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_FILENAME}' not found. Please ensure the LightGBM model is uploaded.")
        return None

# --- 3. PLOTTING FUNCTIONS ---

def create_demand_risk_plot(df_plot, shortage_threshold_col):
    """Plots Demand, Threshold, and highlights shortage hours for clarity."""
    
    if df_plot.empty:
        st.info("No data available for the selected time range and/or time periods.")
        return
        
    # Calculate a boolean column for shortage
    df_plot['Shortage_Risk'] = df_plot[PREDICTION_COL_NAME] > df_plot[shortage_threshold_col]
    
    base = alt.Chart(df_plot).encode( 
        x=alt.X(DATE_COL, title='Forecast Date (Hourly)'),
        tooltip=[
            DATE_COL, 
            alt.Tooltip(PREDICTION_COL_NAME, title="Demand (MW)", format=',.2f'),
            alt.Tooltip(shortage_threshold_col, title="99th Pctl (MW)", format=',.2f'),
            alt.Tooltip(TEMP_CELSIUS_COL, title="Temp (°C)"),
            alt.Tooltip('Time_Period', title="Time Period")
        ]
    )
    
    # 1. Prediction Line (Blue)
    demand_line = base.mark_line(color='#1f77b4').encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)'),
    )
    
    # 2. Dynamic Threshold Line (Red Dashed)
    threshold_line = base.mark_line(color='red', strokeDash=[5, 5]).encode(
        y=alt.Y(shortage_threshold_col),
    )

    # 3. Shortage Highlight (Orange Bars) - Only show when risk is True
    shortage_highlight = base.mark_bar(color='#ff7f0e').encode(
        y=alt.Y(PREDICTION_COL_NAME),
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Shortage_Risk', oneOf=[True])
    )
    
    chart = (demand_line + threshold_line + shortage_highlight).properties(
        title='Hourly Forecast Demand & Shortage Hours vs. Dynamic Threshold'
    )
    
    st.altair_chart(chart, use_container_width=True)


# --- 4. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Dashboard")
    st.title("⚡ Dutch Neighborhood Energy Shortage Predictor")
    st.markdown("---")

    data = load_data()
    model = load_model()
    
    if model is None:
        return
    
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    temp_col_sanitized = sanitize_feature_names([TEMP_COL])[0]
    
    # --- GENERATE FUTURE DATES (Jul 1 - Dec 31, 2025) ---
    future_start_date = data.index[-1] + pd.Timedelta(hours=1) 
    future_end_date = datetime(2025, 12, 31, 23, 0, 0, tzinfo=future_start_date.tzinfo)
    
    future_dates = pd.date_range(start=future_start_date, end=future_end_date, freq='H', name=DATE_COL)
    future_df = pd.DataFrame(index=future_dates)
    
    # Create base time features on the full 6-month period
    future_df = create_features(future_df)

    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: SCENARIO SELECTION AND CALENDAR INPUT ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Forecast Scenario Controls")
    
    # 1. Temperature Scenario Selection
    selected_scenario = st.sidebar.selectbox(
        "1. Select Temperature Scenario:",
        options=list(SCENARIO_MAP.keys()),
        index=1, 
        help="Select a fixed temperature profile for the entire forecast period to test scenarios."
    )
    temp_forecast_celsius = SCENARIO_MAP[selected_scenario]
    
    
    # 2. Date Range Calendar Input
    full_date_range = future_df.index.normalize().unique()
    
    default_start_date = full_date_range[-30].date()
    default_end_date = full_date_range[-1].date()
    
    selected_dates = st.sidebar.date_input(
        "2. Select Forecast Display Range (Calendar):",
        value=(default_start_date, default_end_date),
        min_value=full_date_range.min().date(),
        max_value=full_date_range.max().date(),
        help="Use the calendar to choose the start and end dates to zoom into the results."
    )
    
    # Process selected dates (with timezone fix)
    if len(selected_dates) == 2:
        start_date_filter = pd.to_datetime(selected_dates[0]).tz_localize('UTC').normalize()
        end_date_filter = (pd.to_datetime(selected_dates[1]).tz_localize('UTC').normalize() + pd.Timedelta(hours=23))
        
        # RESTORED: Sidebar info box
        st.sidebar.info(f"Scenario: **{selected_scenario} ({temp_forecast_celsius:.1f}°C)**\n\nDisplaying: **{start_date_filter.strftime('%b %d')}** to **{end_date_filter.strftime('%b %d')}**")
    else:
        st.error("Please select both a start and end date from the calendar.")
        return

    # 3. Time of Day Filter (RESTORED VERBOSE NAMES)
    st.sidebar.markdown("---")
    st.sidebar.header("Display Filtering")
    
    TIME_PERIODS = [
        'Evening (17:00 - 23:59)',
        'Morning (06:00 - 11:59)',
        'Noon (12:00 - 16:59)',
        'Midnight (00:00 - 05:59)',
    ]

    selected_periods = st.sidebar.multiselect(
        "3. Filter by Time of Day:",
        options=TIME_PERIODS,
        default=TIME_PERIODS, # Default to showing all
        help="Filter the main graph to focus on peak or off-peak hours."
    )
    
    st.markdown("---")
    
    # Apply temperature settings to future_df
    temp_0_1_degrees = int(temp_forecast_celsius * 10) 
    future_df[temp_col_sanitized] = temp_0_1_degrees 
    future_df[TEMP_CELSIUS_COL] = temp_forecast_celsius
    
    # Initialize the Target Column (needed for lag/rolling concat)
    future_df[target_col_sanitized] = np.nan
    
    # ----------------------------------------------------------------------
    # --- FEATURE ENGINEERING & PREDICTION (Full 6-Months) ---
    # ----------------------------------------------------------------------
    
    last_hist = data.tail(1)
    expected_model_features = list(model.feature_name_)
    
    ADVANCED_LAG_ROLLING_COLS = [
        'Demand_MW_lag24', 'Demand_MW_lag48', 'Demand_MW_roll72', 
        'temp_lag24', 'temp_roll72', 'temp_roll168'
    ]

    # --- 1. Impute Base and One-Hot Encoded (OHE) Features ---
    for col in expected_model_features:
        if col not in future_df.columns:
            
            if col in ADVANCED_LAG_ROLLING_COLS:
                continue
            
            if any(cat in col for cat in ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']):
                if col in last_hist.columns and last_hist[col].iloc[0] == 1:
                    future_df[col] = 1
                else:
                    future_df[col] = 0
            elif col in last_hist.columns:
                future_df[col] = last_hist[col].iloc[0]
            elif col == 'index':
                future_df[col] = np.arange(len(data), len(data) + len(future_df)) + 1
            else:
                future_df[col] = 0 
    
    # --- 2. Calculate Advanced Lag/Rolling Features ---
    
    lag_cols = [target_col_sanitized, temp_col_sanitized]
    lag_df = pd.concat([data[lag_cols], future_df[lag_cols]]) 
    
    lag_df['Demand_MW_lag24'] = lag_df[target_col_sanitized].shift(24)
    lag_df['Demand_MW_lag48'] = lag_df[target_col_sanitized].shift(48)
    lag_df['Demand_MW_roll72'] = lag_df[target_col_sanitized].shift(1).rolling(72).mean()

    lag_df['temp_lag24'] = lag_df[temp_col_sanitized].shift(24)
    lag_df['temp_roll72'] = lag_df[temp_col_sanitized].shift(1).rolling(72).mean()
    lag_df['temp_roll168'] = lag_df[temp_col_sanitized].shift(1).rolling(168).mean()
    
    lag_features = lag_df.columns.difference(lag_cols)
    future_df = future_df.join(lag_df[lag_features].tail(len(future_df)))
    
    # --- 3. Final Prediction (Full 6-Months) ---
    
    future_df = future_df.fillna(future_df.median())
    model_features = expected_model_features

    future_df[PREDICTION_COL_NAME] = model.predict(future_df[model_features].astype(float))
    
    # --- DASHBOARD LAYOUT AND METRICS ---
    
    st.subheader("1. Core Risk Metrics & Shortage Analysis")
    
    # Calculate HOURLY 99th Percentile Shortage Threshold
    hourly_threshold_df = data.groupby('hour')[target_col_sanitized].quantile(0.99).reset_index()
    hourly_threshold_df.columns = ['hour', 'Shortage_Threshold_MW']
    
    # Prepare the full 6-month prediction DataFrame
    df_full_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    df_full_plot = df_full_plot.reset_index(names=[DATE_COL]) 
    df_full_plot = df_full_plot.merge(hourly_threshold_df, on='hour', how='left')
    
    # Add Time_Period column for filtering
    df_full_plot['Time_Period'] = df_full_plot['hour'].apply(map_hour_to_period)
    
    # Filter the DataFrame based on the Date Calendar and Time of Day selection
    df_plot = df_full_plot[
        (df_full_plot[DATE_COL] >= start_date_filter) & 
        (df_full_plot[DATE_COL] <= end_date_filter) &
        (df_full_plot['Time_Period'].isin(selected_periods)) # Apply Time of Day filter
    ].copy()
    
    # --- RISK ANALYSIS using the FILTERED (displayed) data ---
    
    max_shortage_threshold = hourly_threshold_df['Shortage_Threshold_MW'].max()

    if df_plot.empty:
         st.error("No data points for the selected date range and time periods. Please adjust your filters.")
         return
         
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    peak_row = df_plot.loc[df_plot[PREDICTION_COL_NAME].idxmax()]
    peak_time_full = peak_row[DATE_COL].strftime('%Y-%m-%d %H:%M')
    peak_temp = peak_row[TEMP_CELSIUS_COL]
    
    shortage_hours = df_plot[df_plot[PREDICTION_COL_NAME] > df_plot['Shortage_Threshold_MW']]
    
    if peak_demand > CAPACITY_THRESHOLD:
        risk_level = "CRITICAL"
        delta_val = peak_demand - CAPACITY_THRESHOLD
        delta = f"↑ {delta_val:,.2f} MW above Hard Capacity"
    elif not shortage_hours.empty:
        risk_level = "HIGH"
        max_shortage_delta = (shortage_hours[PREDICTION_COL_NAME] - shortage_hours['Shortage_Threshold_MW']).max()
        delta = f"↑ {max_shortage_delta:,.2f} MW above Hourly 99th Pctl"
    else:
        risk_level = "LOW"
        delta_val = max_shortage_threshold - peak_demand
        delta = f"↓ {delta_val:,.2f} MW below Max 99th Pctl"
        
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Shortage Risk Score (Filtered View)", 
            value=risk_level, 
            delta=f"Hours above 99th Pctl: {len(shortage_hours)}",
            delta_color="off" 
        )
    with col2:
        st.metric(
            label="Peak Predicted Demand (MW)", 
            value=f"{peak_demand:,.2f}", 
            delta=delta,
            delta_color="normal"
        )
    with col3:
        st.metric(
            label="Forecast Temperature at Peak (°C)", 
            value=f"{peak_temp:.1f}°C", 
            delta=f"Time: {peak_time_full}", 
            delta_color="off"
        )
        
    st.markdown("---")
    
    # --- DASHBOARD PLOTS ---
    
    st.subheader("2. Hourly Energy Demand Forecast and Shortage Risk")
    
    # Only keep the Demand Risk plot
    create_demand_risk_plot(df_plot, 'Shortage_Threshold_MW')


# Execute the main function
if __name__ == "__main__":
    main()