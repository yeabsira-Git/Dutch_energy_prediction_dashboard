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
MA_WINDOW = 168 # 7 days * 24 hours
MA_ALERT_BUFFER = 500 # Buffer (MW) added to the Moving Average
GLOBAL_RISK_THRESHOLD = 15500 # A high statistical peak, used as a fixed red line on the plot

# Use the time periods as the options list
TIME_OF_DAY_OPTIONS = ['Morning', 'Noon', 'Evening', 'Midnight']

# --- HELPER FUNCTIONS FOR NEW FEATURES ---

def get_temp_category(temp_c):
    """Categorizes temperature based on user's EDA findings (in Celsius)."""
    if temp_c <= 10:
        return 'Cold (Peak Noon)'
    elif temp_c <= 20:
        return 'Mild (Peak Noon)'
    elif temp_c <= 25:
        return 'Warm (Peak Evening)'
    else:
        return 'Summer (Peak Evening)'

# --- DATA LOADING AND PREDICTION ---

@st.cache_data
def load_data_and_predict():
    """
    Loads historical data, engineers features, loads the LightGBM model, 
    and generates the Predicted_Demand column, aligning features robustly.
    """
    data_file_path = 'cleaned_energy_weather_data(1).csv' 
    model_file_path = MODEL_FILENAME
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_file_path, parse_dates=[DATE_COL], encoding='latin1')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # 2. Basic Feature Engineering (Crucial: Temperature conversion)
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10.0
    df['hour'] = df[DATE_COL].dt.hour
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Handle required lag/roll features (Mimicking the model's training features)
    # The joblib model snippet shows these features are needed:
    # temp_lag24, temp_roll72, temp_roll168, Demand_MW_lag24, Demand_MW_lag48, Demand_MW_roll72
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
        st.error(f"Error loading model: {e}")
        st.stop()
        
    # 5. Select Features (Ensure columns match the model's expected features)
    model_features = model.feature_name_
    # FIX: Only use features present in the dataframe to avoid errors
    features_to_predict = [f for f in model_features if f in df.columns]
    
    # 6. Predict (Handle NaNs introduced by lags by filling/dropping as needed for prediction)
    # NOTE: The model uses historical and future weather data, so for a real future forecast, 
    # we'd need forecasted weather. Here we use the actual future weather data in the CSV 
    # for a "simulated" accurate prediction.
    df_predict = df.copy().dropna(subset=features_to_predict)

    # Make prediction
    X_predict = df_predict[features_to_predict]
    y_pred = model.predict(X_predict)
    
    # Merge prediction back to original dataframe (only for rows where prediction was possible)
    df[PREDICTION_COL_NAME] = np.nan
    df.loc[df_predict.index, PREDICTION_COL_NAME] = y_pred
    
    # Add temperature category
    df['Weather_Category'] = df[TEMP_CELSIUS_COL].apply(get_temp_category)

    # Final cleanup and return
    return df

# --- VISUALIZATION FUNCTIONS ---

def create_demand_forecast_plot(df: pd.DataFrame):
    """Creates the standard time-series demand and forecast plot."""
    df_long = df.melt(
        id_vars=[DATE_COL], 
        value_vars=[ORIGINAL_TARGET_COL, PREDICTION_COL_NAME], 
        var_name='Type', 
        value_name='Demand (MW)'
    )
    
    # Base chart
    base = alt.Chart(df_long).encode(
        x=alt.X(DATE_COL, title="Date and Time (UTC)"),
        y=alt.Y('Demand (MW)', title="Energy Demand (MW)"),
        tooltip=[DATE_COL, 'Demand (MW)', 'Type']
    ).properties(
        height=400
    )

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
    """Creates a plot of demand vs. hour of day, segmented by temperature category."""
    
    # Group by hour and temperature category to show the average predicted demand profile
    df_hourly = df.groupby(['hour', 'Weather_Category'])[PREDICTION_COL_NAME].mean().reset_index(name='Avg Predicted Demand (MW)')
    
    # Determine the overall dominant category for the title
    dominant_category = df['Weather_Category'].mode().iloc[0]
    
    chart = alt.Chart(df_hourly).mark_line(point=True).encode(
        x=alt.X('hour', title='Hour of Day (0-23)'),
        y=alt.Y('Avg Predicted Demand (MW)'),
        color=alt.Color('Weather_Category', title='Weather Category'),
        tooltip=['hour', 'Weather_Category', alt.Tooltip('Avg Predicted Demand (MW)', format='.0f')]
    ).properties(
        title=f"Predicted Hourly Demand Profile by Weather (Dominant: {dominant_category})"
    )
    
    st.altair_chart(chart, use_container_width=True)

def display_summary_kpis(df: pd.DataFrame):
    """Displays the predicted peak and low demand for the period."""
    predicted_peak = df[PREDICTION_COL_NAME].max()
    predicted_low = df[PREDICTION_COL_NAME].min()
    
    peak_time = df.loc[df[PREDICTION_COL_NAME].idxmax(), DATE_COL].strftime('%Y-%m-%d %H:%M')
    low_time = df.loc[df[PREDICTION_COL_NAME].idxmin(), DATE_COL].strftime('%Y-%m-%d %H:%M')
    
    st.subheader("2. Predicted Demand Peak & Low")
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
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Prototype (Dutch Neighborhoods)")
    st.title("💡 Energy Demand Prediction Prototype")
    st.subheader(f"Project: Early Prediction of Energy Shortages in Dutch Neighborhoods ({datetime.now().strftime('%Y-%m-%d')})")
    
    df_full = load_data_and_predict()
    
    # ----------------------------------------------------------------------
    # --- SIDEBAR CONTROLS ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Controls & Settings")
    
    min_date = df_full[DATE_COL].min().date()
    max_date = df_full[DATE_COL].max().date()
    default_start = max_date - timedelta(days=7) # Default to the last 7 days for a quick view
    
    # Date Slider/Picker for Time Period/Forecast Horizon
    selected_date_range = st.sidebar.slider(
        "📅 Time Period & Forecast Horizon (Days)",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )

    # Simple slider to represent "Forecast Horizon Days" control
    # Although the date range handles the period, an explicit "days" input is often requested.
    # Here, we show the duration of the selected range.
    forecast_days = (selected_date_range[1] - selected_date_range[0]).days
    st.sidebar.metric("Forecast Horizon Days", f"{forecast_days} days")
    
    # Handle the timezone awareness of the DataFrame column
    start_dt = datetime.combine(selected_date_range[0], datetime.min.time(), tzinfo=df_full[DATE_COL].dt.tz).tz_convert('UTC')
    end_dt = datetime.combine(selected_date_range[1], datetime.max.time(), tzinfo=df_full[DATE_COL].dt.tz).tz_convert('UTC')

    # Filter the DataFrame based on the selected date range
    df_filtered = df_full[
        (df_full[DATE_COL] >= start_dt) & 
        (df_full[DATE_COL] <= end_dt)
    ].copy()
    
    if df_filtered.empty:
        st.warning("No data found for the selected date range. Please adjust the slider.")
        return

    # ----------------------------------------------------------------------
    # --- DASHBOARD PLOTS AND KPIS ---
    # ----------------------------------------------------------------------
    
    # FEATURE 1: Display Predicted Peak and Low Demand
    display_summary_kpis(df_filtered)

    # FEATURE 2: Standard Time Series Plot
    st.subheader(f"3. Energy Demand Over Time (Viewing {selected_date_range[0]} to {selected_date_range[1]})")
    create_demand_forecast_plot(df_filtered)

    st.markdown("---")
    
    # FEATURE 3: Demand vs Time (Hour of Day) reflecting EDA findings
    st.subheader("4. Hourly Demand Profile: Visualizing Temperature-Based Peak Shift")
    st.markdown("""
        This plot shows the **average predicted demand profile by hour** for the selected period,
        highlighting the shift in peak demand based on the dominant weather category: 
        **Noon** peak for Cold/Mild vs. **Evening** peak for Warm/Summer.
    """)
    create_hourly_profile_plot(df_filtered)


# Execute the main application
if __name__ == "__main__":
    main()