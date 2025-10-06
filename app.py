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
TEMP_CELSIUS_COL = 'Temperature_C' # New column for plots
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' 
CAPACITY_THRESHOLD = 15000 # Example capacity (in MW). Adjust this based on your network's actual limit.

# List of categorical columns identified from the model's expected features
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# --- 1. UTILITY FUNCTIONS ---

# Helper function to sanitize names
def sanitize_feature_names(columns):
    new_cols = []
    for col in columns:
        col = str(col)
        # Remove special characters and replace with underscore
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        col = re.sub(r'^_+|_+$', '', col)
        col = re.sub(r'_{2,}', '_', col)
        new_cols.append(col)
    return new_cols

# Feature Engineering Function
@st.cache_data
def create_features(df):
    
    # FIX for KeyError: Only set the index if the date column exists (used for CSV data loading)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True) 
        df = df.set_index(DATE_COL)
        
    # 1. Temporal Features (Crucial for Time-of-Day Analysis)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # 2. Temperature Conversion 
    if TEMP_COL in df.columns:
        df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10
    
    # 3. Lag Features (Used only for historical data initially)
    if TARGET_COL in df.columns:
        df['lag_24'] = df[TARGET_COL].shift(24)
    
    return df

# --- 2. DATA/MODEL LOADING ---

@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_energy_weather_data(1).csv')
    data = create_features(data)
    
    # Sanitize columns immediately after feature creation
    data.columns = sanitize_feature_names(data.columns)
    
    # Fill missing lag values with the last available historical value
    data = data.ffill()
    return data

@st.cache_resource
def load_model():
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
    
    # Calculate the average demand for each temperature bin
    temp_bins = pd.cut(data_hist[TEMP_CELSIUS_COL], bins=np.arange(-20, 45, 1), include_lowest=True)
    temp_demand = data_hist.groupby(temp_bins)['Demand_MW'].mean().reset_index()
    temp_demand[TEMP_CELSIUS_COL] = temp_demand[temp_bins.name].apply(lambda x: x.mid)

    # Find the predicted peak demand and its associated temperature
    peak_pred = df_plot.loc[df_plot['Predicted_Demand'].idxmax()]
    
    # Create the Altair chart
    chart_hist = alt.Chart(temp_demand).mark_line(point=True).encode(
        x=alt.X(TEMP_CELSIUS_COL, title='Temperature (°C)'),
        y=alt.Y('Demand_MW', title='Avg. Demand (MW)'),
        tooltip=[alt.Tooltip(TEMP_CELSIUS_COL, title="Temp"), alt.Tooltip('Demand_MW', title="Avg. Demand")]
    ).properties(title='Historical U-Curve: Demand vs. Temperature')

    # Add a layer for the predicted peak point
    chart_peak = alt.Chart(pd.DataFrame({
        TEMP_CELSIUS_COL: [peak_pred[TEMP_CELSIUS_COL]],
        'Predicted_Demand': [peak_pred['Predicted_Demand']]
    })).mark_point(
        color='red',
        size=100
    ).encode(
        x=alt.X(TEMP_CELSIUS_COL),
        y=alt.Y('Predicted_Demand'),
        tooltip=[alt.Tooltip(TEMP_CELSIUS_COL, title="Peak Temp"), alt.Tooltip('Predicted_Demand', title="Peak Demand")]
    )
    
    st.altair_chart(chart_hist + chart_peak, use_container_width=True)

def create_dual_axis_forecast(df_plot, shortage_threshold):
    """Plots the main forecast with demand and temperature on dual axes."""
    
    base = alt.Chart(df_plot.reset_index()).encode(
        x=alt.X(DATE_COL, title='Forecast Hour')
    )

    demand_line = base.mark_line(color='#1f77b4').encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)', axis=alt.Axis(titleColor='#1f77b4')),
        tooltip=[DATE_COL, alt.Tooltip(PREDICTION_COL_NAME, title="Demand")]
    )

    threshold_line = base.mark_rule(color='red', strokeDash=[5, 5]).encode(
        y=alt.YDatum(shortage_threshold),
        tooltip=[alt.Tooltip(f'Shortage Threshold ({shortage_threshold:,.0f} MW)', title='Threshold')]
    )

    temp_line = base.mark_line(color='#ff7f0e').encode(
        y=alt.Y(TEMP_CELSIUS_COL, title='Temperature (°C)', axis=alt.Axis(titleColor='#ff7f0e')),
        tooltip=[DATE_COL, alt.Tooltip(TEMP_CELSIUS_COL, title="Temp")]
    )
    
    chart = alt.layer(demand_line, threshold_line, temp_line).resolve_scale(
        y='independent' 
    ).properties(
        title='24-Hour Forecast: Demand (Blue) & Temperature (Orange)'
    )
    
    st.altair_chart(chart, use_container_width=True)

def create_time_of_day_variance_plot(data_hist, df_plot):
    """Compares the predicted hourly demand profile against the historical average profile."""
    
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    
    # 1. Calculate Historical Average Profile
    avg_profile = data_hist.groupby('hour')[target_col_sanitized].mean().reset_index()
    avg_profile.columns = ['hour', 'Average_Demand']
    
    # 2. Extract Forecast Profile
    forecast_profile = df_plot.reset_index()
    forecast_profile = forecast_profile.groupby('hour')[PREDICTION_COL_NAME].mean().reset_index()
    forecast_profile.columns = ['hour', 'Predicted_Demand']
    
    # 3. Merge for Plotting
    plot_df = pd.merge(avg_profile, forecast_profile, on='hour', how='inner')
    
    # 4. Create Altair chart
    chart = alt.Chart(plot_df).encode(
        x=alt.X('hour', title='Hour of Day (0-23)', scale=alt.Scale(domain=[0, 23]))
    )
    
    # Average line
    avg_line = chart.mark_line(color='grey', strokeDash=[3, 3]).encode(
        y=alt.Y('Average_Demand', title='Demand (MW)'),
        tooltip=['hour', 'Average_Demand']
    )
    
    # Predicted line
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
    
    # Sanitize column names for use after loading
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    temp_col_sanitized = sanitize_feature_names([TEMP_COL])[0]

    # --- SIMULATE NEXT 24-HOUR FORECAST ---
    
    # Get the last 24 hours of actual data
    last_actuals = data[target_col_sanitized].tail(24)
    
    # --- SIMULATE FUTURE DATA FOR PREDICTION (Mocking future weather) ---
    future_start_date = data.index[-1] + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=future_start_date, periods=24, freq='H', name=DATE_COL)
    future_df = pd.DataFrame(index=future_dates)
    
    # Create temporal features for the future data
    future_df = create_features(future_df)
    
    # Mocking Temperature (Cold Snap Example)
    temp_shift = (future_df.index.hour - 6) * (2 * np.pi / 24)
    temp_forecast = (np.cos(temp_shift) * 30) + 20 # Sinusoidal temp swing
    future_df[temp_col_sanitized] = (temp_forecast * 10).astype(int) 
    future_df[TEMP_CELSIUS_COL] = future_df[temp_col_sanitized] / 10 
    
    # Fill in required categorical columns
    for col in CATEGORICAL_COLS:
        col_sanitized = sanitize_feature_names([col])[0]
        if col_sanitized in data.columns:
            mode_val = data[col_sanitized].mode()[0]
            future_df[col_sanitized] = mode_val 

    # --- FEATURE VALIDATION AND PREPARATION (Crucial for LightGBM Fix) ---
    
    expected_model_features = list(model.feature_name_)
    
    # 1. Ensure the 'lag_24' feature is correctly populated
    # We use the last 24 actual values for the first 24 future lags.
    # This aligns the past 24 actual demands with the future 24 forecast hours.
    future_df['lag_24'] = last_actuals.values[:len(future_df)]
    
    # 2. Final check: Ensure all expected features are present
    missing_features = [col for col in expected_model_features if col not in future_df.columns]
    
    if missing_features:
        st.error(f"FATAL ERROR: The model is missing required features in the forecast data: {missing_features}. Cannot predict.")
        return # Stop execution if features are missing
    
    # 3. Select the final feature set for prediction
    model_features = expected_model_features

    # --- RUN PREDICTION ---
    future_df[PREDICTION_COL_NAME] = model.predict(future_df[model_features])

    # --- DASHBOARD LAYOUT AND METRICS ---
    
    st.subheader("1. Core Risk Metrics & Shortage Analysis")
    
    shortage_threshold = data[target_col_sanitized].quantile(0.99)
    
    df_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    
    peak_row = df_plot.loc[df_plot[PREDICTION_COL_NAME].idxmax()]
    peak_time = peak_row.name.strftime('%H:%M')
    peak_temp = peak_row[TEMP_CELSIUS_COL]
    
    # Determine the status and delta for the main risk score
    if peak_demand > CAPACITY_THRESHOLD:
        risk_level = "CRITICAL"
        risk_color = "red"
        delta = f"↑ {peak_demand - CAPACITY_THRESHOLD:,.2f} MW above Capacity"
    elif peak_demand > shortage_threshold:
        risk_level = "HIGH"
        risk_color = "orange"
        delta = f"↑ {peak_demand - shortage_threshold:,.2f} MW above 99th Pctl"
    else:
        risk_level = "LOW"
        risk_color = "green"
        delta = f"↓ {shortage_threshold - peak_demand:,.2f} MW below 99th Pctl"
        
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Shortage Risk Score (24H)", 
            value=risk_level, 
            delta=f"Peak Time: {peak_time}",
            delta_color=risk_color if risk_color in ["red", "orange"] else "normal"
        )
    with col2:
        st.metric(
            label="Peak Predicted Demand (MW)", 
            value=f"{peak_demand:,.2f}", 
            delta=delta,
            delta_color=risk_color
        )
    with col3:
        st.metric(
            label="Predicted Temperature at Peak (°C)", 
            value=f"{peak_temp:.1f}°C", 
            delta=f"Time: {peak_time}", 
            delta_color="normal"
        )
        
    st.markdown("---")
    
    # --- DASHBOARD REFINEMENT: EXPLAINABLE PLOTS ---
    
    st.subheader("2. Explainable Forecast Drivers")
    
    st.markdown("#### 2.1. Dual-Axis Forecast: Demand vs. Temperature")
    create_dual_axis_forecast(df_plot, shortage_threshold)

    
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