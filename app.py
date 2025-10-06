import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
import lightgbm as lgb
import altair as alt # Import Altair for advanced plotting
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' # New column for plots
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' # Added constant for prediction column
CAPACITY_THRESHOLD = 15000 # Example capacity (in MW). Adjust this based on your network's actual limit.

# List of categorical columns identified from the model's expected features
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# --- 1. UTILITY FUNCTIONS (Modified) ---

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

# Feature Engineering Function (FIXED KEY ERROR)
@st.cache_data
def create_features(df):
    
    # --- FIX START ---
    # Only set the index if the date column exists (used for CSV data loading)
    # The future_df already has the date in the index, so this step is skipped.
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True) # Added utc=True for safety
        df = df.set_index(DATE_COL)
    # --- FIX END ---
    
    # 1. Temporal Features (Crucial for Time-of-Day Analysis)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # 2. Temperature Conversion (Crucial for Explainability)
    if TEMP_COL in df.columns:
        # Temperature is in 0.1 degrees Celsius, convert to standard Celsius
        df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10
    
    # 3. Lag Features (If applicable, kept simple here)
    if TARGET_COL in df.columns:
        df['lag_24'] = df[TARGET_COL].shift(24)
    
    return df

# --- 2. DATA/MODEL LOADING ---

@st.cache_data
def load_data():
    # Load historical data (assuming it's in the same directory)
    data = pd.read_csv('cleaned_energy_weather_data(1).csv')
    data = create_features(data)
    # Ensure all features have clean names before model fitting/prediction
    data.columns = sanitize_feature_names(data.columns)
    
    # Handle the fact that some weather data might be missing for the first row or two of the future data
    # We will fill missing lag values with the last available historical value
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

# --- 3. PLOTTING FUNCTIONS (The Core Refinement) ---

def create_u_curve_plot(data_hist, df_plot):
    """
    Plots the historical Demand vs Temperature (U-Curve) and highlights the peak prediction.
    """
    
    # Ensure data is ready for plotting
    data_hist = data_hist.rename(columns={TARGET_COL: 'Demand_MW'})
    df_plot = df_plot.rename(columns={PREDICTION_COL_NAME: 'Predicted_Demand'})
    
    # Calculate the average demand for each temperature bin
    temp_bins = pd.cut(data_hist[TEMP_CELSIUS_COL], bins=np.arange(-20, 45, 1), include_lowest=True)
    temp_demand = data_hist.groupby(temp_bins)['Demand_MW'].mean().reset_index()
    temp_demand['Temperature_C'] = temp_demand[TEMP_CELSIUS_COL].apply(lambda x: x.mid)

    # Find the predicted peak demand and its associated temperature
    peak_pred = df_plot.loc[df_plot['Predicted_Demand'].idxmax()]
    
    # Create the Altair chart
    chart_hist = alt.Chart(temp_demand).mark_line(point=True).encode(
        x=alt.X(TEMP_CELSIUS_COL, title='Temperature (°C)'),
        y=alt.Y('Demand_MW', title='Avg. Demand (MW)'),
        tooltip=[TEMP_CELSIUS_COL, 'Demand_MW']
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
    """
    Plots the main forecast with demand and temperature on dual axes.
    """
    # Create a base chart
    base = alt.Chart(df_plot.reset_index()).encode(
        x=alt.X(DATE_COL, title='Forecast Hour')
    )

    # Demand line and Shortage Zone
    demand_line = base.mark_line(color='#1f77b4').encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)', axis=alt.Axis(titleColor='#1f77b4')),
        tooltip=[DATE_COL, alt.Tooltip(PREDICTION_COL_NAME, title="Demand")]
    )

    # Shortage threshold line
    threshold_line = base.mark_rule(color='red', strokeDash=[5, 5]).encode(
        y=alt.YDatum(shortage_threshold),
        tooltip=[alt.Tooltip(f'Shortage Threshold ({shortage_threshold:,.0f} MW)', title='Threshold')]
    )

    # Temperature line (on secondary axis)
    temp_line = base.mark_line(color='#ff7f0e').encode(
        y=alt.Y(TEMP_CELSIUS_COL, title='Temperature (°C)', axis=alt.Axis(titleColor='#ff7f0e')),
        tooltip=[DATE_COL, alt.Tooltip(TEMP_CELSIUS_COL, title="Temp")]
    )
    
    # Combine charts using Alt.layer for dual axis and threshold
    chart = alt.layer(demand_line, threshold_line, temp_line).resolve_scale(
        y='independent' # Key for dual-axis: make Y scales independent
    ).properties(
        title='24-Hour Forecast: Demand (Blue) & Temperature (Orange)'
    )
    
    st.altair_chart(chart, use_container_width=True)

def create_time_of_day_variance_plot(data_hist, df_plot):
    """
    Compares the predicted hourly demand profile against the historical average profile.
    """
    
    # 1. Calculate Historical Average Profile
    avg_profile = data_hist.groupby('hour')[TARGET_COL].mean().reset_index()
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


# --- 4. STREAMLIT APP LAYOUT (Modified) ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Dashboard")
    st.title("⚡ Dutch Neighborhood Energy Shortage Predictor")
    st.markdown("---")

    data = load_data()
    model = load_model()
    
    if model is None:
        return

    # --- SIMULATE NEXT 24-HOUR FORECAST ---
    
    # Get the last 24 hours of actual data to use as features for the next 24 hours
    # Ensure we are using the TARGET_COL after cleaning up the column names
    target_col_sanitized = sanitize_feature_names([TARGET_COL])[0]
    
    last_actuals = data[target_col_sanitized].tail(24)
    
    # --- SIMULATE FUTURE DATA FOR PREDICTION (Mocking future weather) ---
    
    # Base future data starting from the last date + 1 hour
    future_start_date = data.index[-1] + pd.Timedelta(hours=1)
    future_dates = pd.date_range(start=future_start_date, periods=24, freq='H', name=DATE_COL)
    future_df = pd.DataFrame(index=future_dates)
    
    # Create temporal features for the future data
    future_df = create_features(future_df)
    
    # Mocking Temperature: Simulate a cold snap dropping to -5 C (or -50 in 0.1 degrees) at 6 AM
    
    # Time-based simulation to create a realistic-looking temperature dip around the morning hours
    temp_shift = (future_df.index.hour - 6) * (2 * np.pi / 24)
    temp_forecast = (np.cos(temp_shift) * 30) + 20
    
    # Ensure temperature in 0.1 degrees
    temp_col_sanitized = sanitize_feature_names([TEMP_COL])[0]
    future_df[temp_col_sanitized] = (temp_forecast * 10).astype(int) 
    future_df[TEMP_CELSIUS_COL] = future_df[temp_col_sanitized] / 10 # New Celsius column
    
    # Fill in required dummy/categorical columns with mode of historical data
    for col in CATEGORICAL_COLS:
        col_sanitized = sanitize_feature_names([col])[0]
        if col_sanitized in data.columns:
            # We use the sanitized name for the column we create
            future_df[col_sanitized] = data[col_sanitized].mode()[0] 

    # Select features for prediction (must match model's expected features)
    model_features = [col for col col in model.feature_name_ if col in future_df.columns]
    
    # --- RUN PREDICTION ---
    
    # Fill in lag_24 feature using the last actual demand
    # Simple fill: take the last 24 values of actual data and set them as the lag_24 for the future hours
    future_df['lag_24'] = np.roll(last_actuals.values, -len(future_df))[:len(future_df)]
    
    # Prediction
    future_df[PREDICTION_COL_NAME] = model.predict(future_df[model_features])

    # --- DASHBOARD REFINEMENT: METRICS AND SHORTAGE ALERT ---
    
    st.subheader("1. Core Risk Metrics & Shortage Analysis")
    
    # Calculate Shortage Threshold (99th percentile of historical demand)
    shortage_threshold = data[target_col_sanitized].quantile(0.99)
    
    df_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    
    # Find the hour and temperature corresponding to the peak demand
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
        
    
    # Use three columns for the main metrics
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
    
    # A. Dual-Axis Forecast Chart (The "When" and "How much" driven by "What")
    st.markdown("#### 2.1. Dual-Axis Forecast: Demand vs. Temperature")
    create_dual_axis_forecast(df_plot, shortage_threshold)
    st.markdown("This chart visually connects the predicted demand surges with the forecasted temperature changes, proving the model's core logic.")

    
    colA, colB = st.columns(2)
    
    with colA:
        # B. Historical U-Curve Plot (The "Is this normal?" check)
        st.markdown("#### 2.2. Historical U-Curve Validation")
        create_u_curve_plot(data, df_plot)
        st.markdown("The red dot shows the peak prediction, validating that it falls logically on the historical demand-temperature curve.")
        
    with colB:
        # C. Time-of-Day Variance Plot (The "Behavioral" driver)
        st.markdown("#### 2.3. Hourly Profile Variance")
        create_time_of_day_variance_plot(data, df_plot)
        st.markdown("The difference between the Predicted Profile (Green) and the Historical Average (Grey Dashed) highlights the **excess load** due to current risk factors.")

# Execute the main function
if __name__ == "__main__":
    main()