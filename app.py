
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
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

def create_u_curve_plot(data_hist, df_plot):
    """Plots the historical Demand vs Temperature (U-Curve) and highlights the peak prediction."""
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    pred_col = PREDICTION_COL_NAME
    
    data_hist = data_hist.rename(columns={target_col_sanitized: 'Demand_MW'})
    df_plot = df_plot.rename(columns={pred_col: 'Predicted_Demand'})
    
    temp_bins = pd.cut(data_hist[TEMP_CELSIUS_COL], bins=np.arange(-20, 45, 1), include_lowest=True)
    temp_demand = data_hist.groupby(temp_bins)['Demand_MW'].mean().reset_index()
    temp_demand[TEMP_CELSIUS_COL] = temp_demand[temp_bins.name].apply(lambda x: x.mid)

    peak_pred = df_plot.loc[df_plot['Predicted_Demand'].idxmax()]
    
    chart_hist = alt.Chart(temp_demand).mark_line(point=True).encode(
        x=alt.X(TEMP_CELSIUS_COL, title='Temperature (°C)'),
        y=alt.Y('Demand_MW', title='Avg. Demand (MW)'),
        tooltip=[alt.Tooltip(TEMP_CELSIUS_COL, title="Temp"), alt.Tooltip('Demand_MW', title="Avg. Demand")]
    ).properties(title='Historical U-Curve: Demand vs. Temperature')

    chart_peak = alt.Chart(pd.DataFrame({
        TEMP_CELSIUS_COL: [peak_pred[TEMP_CELSIUS_COL]],
        'Predicted_Demand': [peak_pred['Predicted_Demand']]
    })).mark_point(
        color='red',
        size=100
    ).encode(
        x=alt.X(TEMP_CELSIUS_COL),
        y=alt.Y('Predicted_Demand'),
        tooltip=[alt.Tooltip(TEMP_CELSIUS_COL, title="Forecast Temp"), alt.Tooltip('Predicted_Demand', title="Peak Demand")]
    )
    
    st.altair_chart(chart_hist + chart_peak, use_container_width=True)

def create_dual_axis_forecast_hourly(df_plot, shortage_threshold_col):
    """Plots the main forecast with demand and temperature on dual axes, 
    using a dynamic hourly threshold column."""
    
    # df_plot should already have the DATE_COL as a column from the main function
    
    base = alt.Chart(df_plot).encode( 
        x=alt.X(DATE_COL, title='Forecast Date (Jul 1 - Dec 31, 2025)')
    )

    demand_line = base.mark_line(color='#1f77b4').encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)', axis=alt.Axis(titleColor='#1f77b4')),
        tooltip=[DATE_COL, alt.Tooltip(PREDICTION_COL_NAME, title="Demand")]
    )

    # Threshold line is drawn from the dynamic hourly column
    threshold_line = base.mark_line(color='red', strokeDash=[5, 5]).encode(
        y=alt.Y(shortage_threshold_col, title='Demand (MW)'),
        tooltip=[alt.Tooltip(shortage_threshold_col, title='Hourly 99th Pctl')] 
    )

    temp_line = base.mark_line(color='#ff7f0e').encode(
        y=alt.Y(TEMP_CELSIUS_COL, title='Temperature (°C)', axis=alt.Axis(titleColor='#ff7f0e')),
        tooltip=[DATE_COL, alt.Tooltip(TEMP_CELSIUS_COL, title="Temp")]
    )
    
    chart = alt.layer(demand_line, threshold_line, temp_line).resolve_scale(
        y='independent' 
    ).properties(
        title='6-Month Forecast: Demand (Blue) & Dynamic Hourly Threshold (Red)'
    )
    
    st.altair_chart(chart, use_container_width=True)

def create_time_of_day_variance_plot(data_hist, df_plot):
    """Compares the predicted hourly demand profile against the historical average profile."""
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    
    avg_profile = data_hist.groupby('hour')[target_col_sanitized].mean().reset_index()
    avg_profile.columns = ['hour', 'Average_Demand']
    
    forecast_profile = df_plot.reset_index(names=[DATE_COL])
    forecast_profile = forecast_profile.groupby('hour')[PREDICTION_COL_NAME].mean().reset_index()
    forecast_profile.columns = ['hour', 'Predicted_Demand']
    
    plot_df = pd.merge(avg_profile, forecast_profile, on='hour', how='inner')
    
    chart = alt.Chart(plot_df).encode(
        x=alt.X('hour', title='Hour of Day (0-23)', scale=alt.Scale(domain=[0, 23]))
    )
    
    avg_line = chart.mark_line(color='grey', strokeDash=[3, 3]).encode(
        y=alt.Y('Average_Demand', title='Demand (MW)'),
        tooltip=['hour', 'Average_Demand']
    )
    
    pred_line = chart.mark_line(color='green').encode(
        y=alt.Y('Predicted_Demand', title='Demand (MW)'),
        tooltip=['hour', 'Predicted_Demand']
    )
    
    st.altair_chart(avg_line + pred_line, use_container_width=True)


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
    
    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUT: CONTINUOUS SLIDER ---
    # ----------------------------------------------------------------------
    st.sidebar.header("6-Month Forecast Scenario (Jul 1 - Dec 31, 2025)")
    
    temp_forecast_celsius = st.sidebar.slider(
        "Set Persistent Forecast Temperature (°C) for 6 Months:",
        min_value=-15.0, 
        max_value=35.0,
        value=15.0, # Default to a mild temperature
        step=0.5,
        format='%.1f °C',
        help="Simulate a constant temperature over the entire 6-month period (July-Dec) to observe the long-term predictive response."
    )
    
    st.sidebar.info(f"The demand will be predicted based on a constant **{temp_forecast_celsius:.1f}°C** for the next 6 months.")
    st.markdown("---")
    
    # --- GENERATE FUTURE DATA FOR PREDICTION (July 1, 2025 - Dec 31, 2025) ---
    
    future_start_date = data.index[-1] + pd.Timedelta(hours=1) 
    future_end_date = datetime(2025, 12, 31, 23, 0, 0, tzinfo=future_start_date.tzinfo)
    
    future_dates = pd.date_range(start=future_start_date, end=future_end_date, freq='H', name=DATE_COL)
    future_df = pd.DataFrame(index=future_dates)
    
    # Create base time features
    future_df = create_features(future_df)
    
    temp_0_1_degrees = int(temp_forecast_celsius * 10) 

    # Set the two temperature columns
    future_df[temp_col_sanitized] = temp_0_1_degrees 
    future_df[TEMP_CELSIUS_COL] = temp_forecast_celsius
    
    # Initialize the Target Column (needed for lag/rolling concat)
    future_df[target_col_sanitized] = np.nan
    
    # ----------------------------------------------------------------------
    # --- FEATURE ENGINEERING (Lag/Rolling) ---
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
    
    # --- 3. Final Prediction ---
    
    future_df = future_df.fillna(future_df.median())
    model_features = expected_model_features

    future_df[PREDICTION_COL_NAME] = model.predict(future_df[model_features].astype(float))
    
    # --- DASHBOARD LAYOUT AND METRICS ---
    
    st.subheader("1. Core Risk Metrics & Shortage Analysis")
    
    # ----------------------------------------------------------------------
    # FIX: Calculate HOURLY 99th Percentile Shortage Threshold
    # ----------------------------------------------------------------------
    
    # Calculate the 99th percentile demand for each hour of the day (0-23)
    hourly_threshold_df = data.groupby('hour')[target_col_sanitized].quantile(0.99).reset_index()
    hourly_threshold_df.columns = ['hour', 'Shortage_Threshold_MW']
    
    # Merge the hourly threshold into the forecast DataFrame
    df_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    df_plot = df_plot.reset_index(names=[DATE_COL]) 
    df_plot = df_plot.merge(hourly_threshold_df, on='hour', how='left')
    
    # The shortage threshold is now a dynamic column in df_plot
    shortage_threshold_series = df_plot['Shortage_Threshold_MW']
    
    # Use the max of this threshold for display purposes
    max_shortage_threshold = hourly_threshold_df['Shortage_Threshold_MW'].max()

    # Analysis based on the 6-month prediction
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    
    peak_row = df_plot.loc[df_plot[PREDICTION_COL_NAME].idxmax()]
    peak_time_full = peak_row[DATE_COL].strftime('%Y-%m-%d %H:%M')
    peak_temp = peak_row[TEMP_CELSIUS_COL]
    
    # Risk calculation now compares predicted demand to the dynamic hourly threshold
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
            label="Shortage Risk Score (6-Month Forecast)", 
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
    
    # --- DASHBOARD REFINEMENT: EXPLAINABLE PLOTS ---
    
    st.subheader("2. Explainable Forecast Drivers")
    
    st.markdown("#### 2.1. Long-Term Forecast: Demand vs. Temperature")
    # Call the new hourly plotting function
    create_dual_axis_forecast_hourly(df_plot, 'Shortage_Threshold_MW')

    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("#### 2.2. Historical U-Curve Validation")
        create_u_curve_plot(data, df_plot)
        
    with colB:
        st.markdown("#### 2.3. Hourly Profile Variance")
        create_time_of_day_variance_plot(data, df_plot)


# Execute the main function
if __name__ == "__main__":
    main()