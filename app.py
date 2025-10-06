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
CAPACITY_THRESHOLD = 15000 # Retained for plot display reference

# RELATIVE RISK THRESHOLDS
MA_ALERT_BUFFER = 500 # Buffer (MW) added to the 168-hour Moving Average

# NOTE: CRITICAL PERIODS ARE KEPT FOR REFERENCE, but not used in the simplified risk logic
COLD_MILD_CRITICAL_PERIOD = 'Noon (12:00 - 16:59)'
WARM_SUMMER_CRITICAL_PERIOD = 'Evening (17:00 - 23:59)'

CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# --- SCENARIO MAPPING (Still used for fixed temperature input) ---
SCENARIO_MAP = {
    "1. Cold (0°C - 10°C)": 5.0,     
    "2. Mild (10°C - 20°C)": 15.0,   
    "3. Warm (20°C - 25°C)": 22.5,   
    "4. Summer (> 25°C)": 30.0      
}

# --- UTILITY FUNCTIONS ---

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

# --- DATA/MODEL LOADING ---

@st.cache_data
def load_data():
    """Loads and preprocesses HISTORICAL data."""
    data = pd.read_csv('cleaned_energy_weather_data(1).csv') 
    data = create_features(data)
    
    # Perform One-Hot Encoding on the historical data
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=False)
    
    # Sanitize all columns (including new OHE ones)
    data.columns = sanitize_feature_names(data.columns)
    
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

# --- PLOTTING FUNCTIONS (Restored to Risk Plot) ---

def create_demand_risk_plot(df_plot, shortage_threshold_col, dynamic_threshold_col):
    """Plots Demand, Thresholds, and highlights hours above the statistical threshold."""
    
    if df_plot.empty:
        st.info("No data available for the selected time range.")
        return
        
    # Re-enable risk highlighting for the plot
    df_plot['Shortage_Risk'] = df_plot[PREDICTION_COL_NAME] > df_plot[shortage_threshold_col]
    df_plot['Hard_Capacity_MW'] = CAPACITY_THRESHOLD
    
    base = alt.Chart(df_plot).encode( 
        x=alt.X(DATE_COL, title='Forecast Date (Hourly)'),
        tooltip=[
            DATE_COL, 
            alt.Tooltip(PREDICTION_COL_NAME, title="Demand (MW)", format=',.2f'),
            alt.Tooltip(shortage_threshold_col, title="Statistical Pctl (MW)", format=',.2f'),
            alt.Tooltip('Hard_Capacity_MW', title="Capacity Reference (MW)", format=',.2f'),
            alt.Tooltip(dynamic_threshold_col, title=f"MA + {MA_ALERT_BUFFER} MW", format=',.2f'), 
            alt.Tooltip(TEMP_CELSIUS_COL, title="Temp (°C)"),
        ]
    )
    
    # 1. Prediction Line 
    demand_line = base.mark_line(point=True).encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)'),
        color=alt.value('darkblue'),
        strokeWidth=alt.value(2)
    )
    
    # 2. Global Risk Threshold Line (Red Dashed - STATISTICAL RISK)
    threshold_line = base.mark_line(color='red', strokeDash=[5, 5]).encode(
        y=alt.Y(shortage_threshold_col),
    )

    # 3. Hard Capacity Line (Black Dotted - OPERATIONAL LIMIT)
    capacity_line = base.mark_line(color='black', strokeDash=[1, 1], size=1).encode(
        y=alt.Y('Hard_Capacity_MW'),
    )
    
    # 4. Dynamic Alert Threshold (Purple Dotted - MA + BUFFER)
    dynamic_line = base.mark_line(color='#800080', strokeDash=[3, 3], size=1).encode(
        y=alt.Y(dynamic_threshold_col),
    )

    # 5. Shortage Highlight (Orange Bars) - Only show when risk is True
    shortage_highlight = base.mark_bar(color='#ff7f0e').encode(
        y=alt.Y(PREDICTION_COL_NAME),
    ).transform_filter(
        alt.FieldOneOfPredicate(field='Shortage_Risk', oneOf=[True])
    )
    
    # Combine all lines and marks
    chart = (demand_line + threshold_line + capacity_line + dynamic_line + shortage_highlight).properties(
        title=f'Hourly Energy Demand Forecast (Risk Focused) for {df_plot[TEMP_CELSIUS_COL].iloc[0]:.1f}°C Scenario'
    )
    
    st.altair_chart(chart.interactive(), use_container_width=True)


# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Dashboard")
    st.title("⚡ Dutch Neighborhood Energy Shortage Prediction Dashboard")
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
    
    future_df = create_features(future_df)

    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: SCENARIO SELECTION AND CALENDAR INPUT ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Forecast Scenario Controls")
    
    # 1. Temperature Scenario Selection
    selected_scenario = st.sidebar.selectbox(
        "1. Select Temperature Scenario:",
        options=list(SCENARIO_MAP.keys()),
        index=0, 
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
        
        # Sidebar info box
        st.sidebar.info(f"Scenario: **{selected_scenario} ({temp_forecast_celsius:.1f}°C)**\n\nDisplaying: **{start_date_filter.strftime('%b %d')}** to **{end_date_filter.strftime('%b %d')}**")
    else:
        st.error("Please select both a start and end date from the calendar.")
        return

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
    
    # 2. Calculate Advanced Lag/Rolling Features 
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
    
    # 3. Final Prediction (Full 6-Months) 
    future_df = future_df.fillna(future_df.median())
    model_features = expected_model_features

    future_df[PREDICTION_COL_NAME] = model.predict(future_df[model_features].astype(float))
    
    # ----------------------------------------------------------------------
    # --- RISK CALCULATION (Restored) ---
    # ----------------------------------------------------------------------
    
    # Calculate HOURLY 99th Percentile Shortage Threshold 
    hourly_threshold_df = data.groupby('hour')[target_col_sanitized].quantile(0.99).reset_index()
    hourly_threshold_df.columns = ['hour', 'Shortage_Threshold_MW']
    
    GLOBAL_RISK_THRESHOLD = hourly_threshold_df['Shortage_Threshold_MW'].max()
    
    # Prepare the full 6-month prediction DataFrame
    df_full_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    df_full_plot = df_full_plot.reset_index(names=[DATE_COL]) 
    df_full_plot['Global_Risk_Threshold_MW'] = GLOBAL_RISK_THRESHOLD
    
    # Filter the DataFrame based on the Date Calendar
    df_plot = df_full_plot[
        (df_full_plot[DATE_COL] >= start_date_filter) & 
        (df_full_plot[DATE_COL] <= end_date_filter) 
    ].copy()
    
    # Calculate Dynamic Threshold for the FILTERED (displayed) data
    df_plot['168_hr_MA'] = df_plot[PREDICTION_COL_NAME].rolling(window=168, min_periods=1).mean()
    df_plot['Dynamic_Alert_Threshold'] = df_plot['168_hr_MA'] + MA_ALERT_BUFFER
    
    
    # ----------------------------------------------------------------------
    # --- DASHBOARD LAYOUT AND METRICS (Restored) ---
    # ----------------------------------------------------------------------
    
    st.subheader("1. Core Forecast Risk Metrics & Analysis")
    
    if df_plot.empty:
         st.error("No data points for the selected date range. Please adjust your filters.")
         return
         
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    peak_row = df_plot.loc[df_plot[PREDICTION_COL_NAME].idxmax()]
    peak_time_full = peak_row[DATE_COL].strftime('%Y-%m-%d %H:%M')
    peak_temp = peak_row[TEMP_CELSIUS_COL]
    peak_hour = peak_row['hour']
    dynamic_trigger_value = peak_row['Dynamic_Alert_Threshold']
    
    peak_above_dynamic = peak_demand > dynamic_trigger_value

    # --- SIMPLIFIED RELATIVE RISK LOGIC (HIGH DEMAND / DYNAMIC SPIKE / LOW DEMAND) ---
    
    if peak_demand > GLOBAL_RISK_THRESHOLD:
        risk_level = "HIGH DEMAND"
        delta_val = peak_demand - GLOBAL_RISK_THRESHOLD
        delta = f"↑ {delta_val:,.2f} MW **above Global 99th Pctl!**"
        delta_color = "inverse"
        
    elif peak_above_dynamic:
        risk_level = "DYNAMIC SPIKE"
        delta_val = peak_demand - dynamic_trigger_value
        delta = f"↑ {delta_val:,.2f} MW **above MA (Short-Term Surge)**"
        delta_color = "normal"
    
    else:
        risk_level = "LOW DEMAND"
        margin_below_dynamic = dynamic_trigger_value - peak_demand if dynamic_trigger_value > peak_demand else 0 
        delta = f"Peak is ↓ {margin_below_dynamic:,.2f} MW **below Dynamic Alert Threshold**"
        delta_color = "off"
        
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Forecast Risk Status (Filtered View)", 
            value=risk_level, 
            delta=delta,
            delta_color=delta_color 
        )
    with col2:
        st.metric(
            label="Peak Predicted Demand (MW)", 
            value=f"{peak_demand:,.2f}", 
            delta=f"Peak Hour: {peak_hour:02d}:00",
            delta_color="off"
        )

    st.markdown(f"**Scenario Temperature:** **{peak_temp:.1f}°C** (Fixed for Forecast)")
    st.markdown(f"**Absolute Risk Trigger (99th Pctl):** **{GLOBAL_RISK_THRESHOLD:,.2f} MW** (Red Dashed Line)")
    st.markdown(f"**Relative Spike Trigger (MA + 500 MW):** **{dynamic_trigger_value:,.2f} MW** (Purple Dotted Line)")
    st.markdown("---")
    
    # --- DASHBOARD PLOTS ---
    
    st.subheader("2. Hourly Demand Forecast for Selected Period (Risk Visualization)")
    
    # Call the restored risk plot function
    create_demand_risk_plot(df_plot, 'Global_Risk_Threshold_MW', 'Dynamic_Alert_Threshold')


# Execute the main function
if __name__ == "__main__":
    main()