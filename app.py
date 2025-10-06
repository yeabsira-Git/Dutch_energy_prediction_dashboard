import streamlit as st
import pandas as pd
import numpy as np
import altair as alt 
from datetime import datetime, date, timedelta
import warnings
import joblib
import lightgbm as lgb
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
ORIGINAL_TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' 

# RELATIVE RISK THRESHOLDS - Keeping relevant risk variables for context
GLOBAL_RISK_THRESHOLD = 15500 # A high statistical peak, used as a fixed red line on the plot

# --- HELPER FUNCTIONS FOR NEW FEATURES ---

def get_temp_category(temp_c):
    """Categorizes temperature based on user's EDA findings (in Celsius).
    Cold (<=10) and Mild (10-20) -> Peak Noon
    Warm (20-25) and Summer (>25) -> Peak Evening
    """
    if temp_c <= 10:
        return 'Cold (Peak Noon ≤ 10°C)'
    elif temp_c <= 20:
        return 'Mild (Peak Noon 10-20°C)'
    elif temp_c <= 25:
        return 'Warm (Peak Evening 20-25°C)'
    else:
        return 'Summer (Peak Evening > 25°C)'

# --- DATA LOADING AND PREDICTION ---

@st.cache_data
def load_data_and_predict():
    """
    Loads historical data, engineers features, loads the LightGBM model, 
    and generates the Predicted_Demand column.
    """
    data_file_path = 'cleaned_energy_weather_data(1).csv' 
    model_file_path = MODEL_FILENAME
    
    # 1. Load Data
    try:
        # FIX FOR 'No columns to parse from file' ERROR: 
        # Explicitly set the separator to comma (',') as the file header indicates standard CSV format.
        df = pd.read_csv(data_file_path, parse_dates=[DATE_COL], encoding='latin1', sep=',')
    except Exception as e:
        st.error(f"FATAL DATA ERROR: Could not load data. Check file format or separator (Error: {e}).")
        st.stop()
    
    # Ensure DateUTC is timezone-aware for filtering later
    if df[DATE_COL].dt.tz is None:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize('UTC') 
    
    # 2. Basic Feature Engineering 
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10.0
    df['hour'] = df[DATE_COL].dt.hour
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Handle required lag/roll features (These must match the model's training features)
    df['Demand_MW_lag24'] = df[ORIGINAL_TARGET_COL].shift(24)
    df['Demand_MW_lag48'] = df[ORIGINAL_TARGET_COL].shift(48)
    df['Demand_MW_roll72'] = df[ORIGINAL_TARGET_COL].shift(1).rolling(window=72).mean()
    df['temp_lag24'] = df[TEMP_CELSIUS_COL].shift(24)
    df['temp_roll72'] = df[TEMP_CELSIUS_COL].shift(1).rolling(window=72).mean()
    df['temp_roll168'] = df[TEMP_CELSIUS_COL].shift(1).rolling(window=168).mean()
    
    # 4. Load Model
    try:
        model = joblib.load(model_file_path)
    except Exception as e:
        st.error(f"FATAL MODEL ERROR: Could not load model '{MODEL_FILENAME}'. Ensure file exists and is correct (Error: {e}).")
        st.stop()
        
    # 5. Select Features
    model_features = model.feature_name_
    features_to_predict = [f for f in model_features if f in df.columns]
    
    # 6. Predict
    df_predict = df.copy().dropna(subset=features_to_predict)

    X_predict = df_predict[features_to_predict]
    y_pred = model.predict(X_predict)
    
    df[PREDICTION_COL_NAME] = np.nan
    df.loc[df_predict.index, PREDICTION_COL_NAME] = y_pred
    
    # Add temperature category
    df['Weather_Category'] = df[TEMP_CELSIUS_COL].apply(get_temp_category)

    return df

# --- VISUALIZATION FUNCTIONS ---

def create_demand_forecast_plot(df: pd.DataFrame):
    """Creates the standard time-series demand and forecast plot (Demand vs Time)."""
    df_long = df.melt(
        id_vars=[DATE_COL], 
        value_vars=[ORIGINAL_TARGET_COL, PREDICTION_COL_NAME], 
        var_name='Type', 
        value_name='Demand (MW)'
    ).dropna(subset=['Demand (MW)'])
    
    # Base chart
    base = alt.Chart(df_long).encode(
        x=alt.X(DATE_COL, title="Date and Time (UTC)"),
        y=alt.Y('Demand (MW)', title="Energy Demand (MW)"),
        tooltip=[DATE_COL, 'Demand (MW)', 'Type']
    ).properties(height=400)

    # Actual Demand (Blue solid line)
    actual = base.transform_filter(
        alt.datum.Type == ORIGINAL_TARGET_COL
    ).mark_line(color='blue').properties(title="Demand Forecast (Actual vs. Predicted)")

    # Predicted Demand (Orange dashed line)
    predicted = base.transform_filter(
        alt.datum.Type == PREDICTION_COL_NAME
    ).mark_line(color='orange', strokeDash=[5, 5])

    # Risk Threshold (Red horizontal line)
    risk_line = alt.Chart(pd.DataFrame({'y': [GLOBAL_RISK_THRESHOLD]})).mark_rule(color='red').encode(
        y='y',
        tooltip=alt.Tooltip('y', title='Risk Threshold')
    )
    
    st.altair_chart(actual + predicted + risk_line, use_container_width=True)

def create_hourly_profile_plot(df: pd.DataFrame):
    """Creates a plot of demand vs. hour of day (Time of Day and Demand)."""
    
    # Group by hour and temperature category to show the average predicted demand profile
    df_hourly = df.groupby(['hour', 'Weather_Category'])[PREDICTION_COL_NAME].mean().reset_index(name='Avg Predicted Demand (MW)')
    
    # Determine the overall dominant category for the title
    dominant_category = df['Weather_Category'].mode().iloc[0].split('(')[1].replace(')', '').strip()
    
    chart = alt.Chart(df_hourly).mark_line(point=True).encode(
        x=alt.X('hour', title='Hour of Day (0-23)'),
        y=alt.Y('Avg Predicted Demand (MW)'),
        color=alt.Color('Weather_Category', title='Weather Category'),
        tooltip=['hour', 'Weather_Category', alt.Tooltip('Avg Predicted Demand (MW)', format='.0f')]
    ).properties(
        title=f"Predicted Hourly Demand Profile by Weather (Dominant Temp: {dominant_category})"
    )
    
    st.altair_chart(chart, use_container_width=True)

def display_summary_kpis(df: pd.DataFrame):
    """Displays the predicted peak and low demand for the period."""
    if df[PREDICTION_COL_NAME].empty:
        st.warning("No predictions available for the selected range.")
        return
        
    predicted_peak = df[PREDICTION_COL_NAME].max()
    predicted_low = df[PREDICTION_COL_NAME].min()
    
    peak_time = df.loc[df[PREDICTION_COL_NAME].idxmax(), DATE_COL].strftime('%Y-%m-%d %H:%M UTC')
    low_time = df.loc[df[PREDICTION_COL_NAME].idxmin(), DATE_COL].strftime('%Y-%m-%d %H:%M UTC')
    
    st.subheader("2. Predicted Demand Peak & Low 🎯")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Peak Demand 📈", 
            value=f"{predicted_peak:,.0f} MW", 
            help=f"Occurs at: {peak_time}"
        )
    with col2:
        st.metric(
            label="Predicted Low Demand 📉", 
            value=f"{predicted_low:,.0f} MW", 
            help=f"Occurs at: {low_time}"
        )
    st.markdown("---")

# --- MAIN STREAMLIT APPLICATION ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Prototype")
    st.title("💡 Energy Demand Prediction Prototype")
    st.subheader(f"Project: Early Prediction of Energy Shortages in Dutch Neighborhoods ({datetime.now().strftime('%Y-%m-%d')})")
    
    df_full = load_data_and_predict()
    
    # ----------------------------------------------------------------------
    # --- SIDEBAR CONTROLS (Slider in Calendar and Forecast Horizon Days) ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Controls & Settings")
    
    # Use max_date for the default end point to show a forecast
    max_date_data = df_full[DATE_COL].max().date()
    min_date_data = df_full[DATE_COL].min().date()
    default_start = max_date_data - timedelta(days=7) 
    
    # Date Slider/Picker for Time Period/Forecast Horizon
    selected_date_range = st.sidebar.slider(
        "📅 Time Period & Forecast Horizon (Select a range)",
        min_value=min_date_data,
        max_value=max_date_data,
        value=(default_start, max_date_data),
        format="YYYY-MM-DD"
    )

    forecast_days = (selected_date_range[1] - selected_date_range[0]).days
    st.sidebar.metric("Selected Forecast Horizon", f"{forecast_days} days")
    
    # Convert selected dates back to timezone-aware datetimes for filtering
    start_dt = datetime.combine(selected_date_range[0], datetime.min.time()).replace(tzinfo=df_full[DATE_COL].dt.tz).normalize()
    end_dt = datetime.combine(selected_date_range[1], datetime.max.time()).replace(tzinfo=df_full[DATE_COL].dt.tz)

    # Filter the DataFrame based on the selected date range
    df_filtered = df_full[
        (df_full[DATE_COL] >= start_dt) & 
        (df_full[DATE_COL] <= end_dt)
    ].copy()
    
    if df_filtered.empty:
        st.warning("No data or predictions found for the selected date range. Please adjust the slider.")
        return

    # ----------------------------------------------------------------------
    # --- DASHBOARD PLOTS AND KPIS ---
    # ----------------------------------------------------------------------
    
    # FEATURE 1: Predicted Peak and Low Demand
    display_summary_kpis(df_filtered)

    # FEATURE 2: Demand vs Time (Time-series plot)
    st.subheader(f"3. Energy Demand Over Time (Viewing {selected_date_range[0]} to {selected_date_range[1]})")
    create_demand_forecast_plot(df_filtered)

    st.markdown("---")
    
    # FEATURE 3: Demand vs Time of Day (Hourly Profile reflecting EDA findings)
    st.subheader("4. Hourly Demand Profile: Visualizing Temperature-Based Peak Shift 🌡️")
    st.markdown("""
        This plot shows the **average predicted demand profile by hour (0-23)** for the selected period.
        It directly illustrates the shift identified in your EDA: 
        Peak demand is expected at **Noon** for Cold/Mild weather and in the **Evening** for Warm/Summer weather.
    """)
    create_hourly_profile_plot(df_filtered)


# Execute the main application
if __name__ == "__main__":
    main()