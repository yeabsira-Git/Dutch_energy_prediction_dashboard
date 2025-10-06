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
        # FIX: Using encoding='latin1' and sep=',' to resolve EmptyDataError/ParserError
        df = pd.read_csv(data_file_path, sep=',', encoding='latin1') 
    except FileNotFoundError:
        st.error(f"Data file '{data_file_path}' not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file was found but appears to be empty or unreadable (EmptyDataError).")
        return pd.DataFrame()

    # 2. Basic Feature Engineering (Must match training features)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)
    df['hour'] = df[DATE_COL].dt.hour
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10

    # 3. Time-of-Day Mapping
    def map_custom_time_of_day(hour):
        if 0 <= hour <= 5: return 'Midnight'
        elif 6 <= hour <= 11: return 'Morning'
        elif 12 <= hour <= 16: return 'Noon'
        elif 17 <= hour <= 23: return 'Evening'
        else: return 'Other'
            
    df['Detailed_Time_of_Day'] = df['hour'].apply(map_custom_time_of_day)
    df = df[df['Detailed_Time_of_Day'].isin(TIME_OF_DAY_OPTIONS)].copy()

    # 4. Prepare Features for Model
    FEATURE_COLS = [
        'index', 'Cov_ratio', 'Wind Direction (degrees)', 'Hourly Mean Wind Speed (0.1 m/s)', 
        'Mean Wind Speed (0.1 m/s)', 'Maximum Wind Gust (0.1 m/s)', 
        'Dew Point Temperature (0.1 degrees Celsius)', 'Sunshine Duration (0.1 hours)', 
        'Precipitation Duration (0.1 hours)', 'Hourly Precipitation Amount (0.1 mm)', 
        'Air Pressure (0.1 hPa)', 'Horizontal Visibility', 'Cloud Cover (octants)', 
        'Relative Atmospheric Humidity (%)', 'Temperature (0.1 degrees Celsius)', 
        'Global_Radiation_kW/m2', 'hour', TEMP_CELSIUS_COL
    ]
    
    df_features = df[FEATURE_COLS].copy()
    
    # Handle Categorical Columns with One-Hot Encoding (OHE)
    df_features = pd.concat([df_features, pd.get_dummies(df['CountryCode'], prefix='CountryCode', drop_first=False)], axis=1)
    df_features = pd.concat([df_features, pd.get_dummies(df['Detailed_Time_of_Day'], prefix='Detailed_Time_of_Day', drop_first=False)], axis=1)
    
    # 5. Load Model and Predict
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_file_path}' not found. Cannot generate predictions.")
        df[PREDICTION_COL_NAME] = np.nan # Add empty prediction column
        return df

    # Robust Feature Alignment to resolve KeyError
    try:
        model_feature_names = list(model.feature_name_)
        missing_cols = set(model_feature_names) - set(df_features.columns)
        for col in missing_cols:
            df_features[col] = 0
            
        extra_cols = set(df_features.columns) - set(model_feature_names)
        df_features = df_features.drop(columns=list(extra_cols))
        X_predict = df_features[model_feature_names]

    except AttributeError:
        st.error("Model feature names not found. Skipping prediction.")
        df[PREDICTION_COL_NAME] = np.nan
        return df
        
    # Predict
    df[PREDICTION_COL_NAME] = model.predict(X_predict)
    
    return df

# --- PLOTTING FUNCTION (Time Series) ---

@st.cache_data
def create_demand_forecast_plot(df_filtered):
    """
    Generates a time-series line chart showing Historical Demand, Predicted Demand, 
    and relative/absolute risk thresholds.
    """
    
    if df_filtered.empty:
        st.warning("No data available for the selected period.")
        return

    # Calculate 168-hour Moving Average (MA) on the historical Demand_MW column
    # Use rolling window, center=False (trailing average)
    df_filtered['MA_Demand'] = df_filtered[ORIGINAL_TARGET_COL].rolling(window=MA_WINDOW, min_periods=1).mean()
    
    # Calculate Dynamic Alert Threshold: MA + 500 MW
    df_filtered['Dynamic_Alert'] = df_filtered['MA_Demand'] + MA_ALERT_BUFFER
    
    # Melt the DataFrame for Altair plotting (Historical and Predicted)
    df_plot = df_filtered.melt(
        id_vars=[DATE_COL, 'MA_Demand', 'Dynamic_Alert', TEMP_CELSIUS_COL, 'Detailed_Time_of_Day'],
        value_vars=[ORIGINAL_TARGET_COL, PREDICTION_COL_NAME],
        var_name='Type',
        value_name='Demand_MW'
    )
    
    # Base chart setup
    base = alt.Chart(df_plot).encode(
        x=alt.X(DATE_COL, title='Date and Time (Hour)'),
        y=alt.Y('Demand_MW', title='Demand (MW)', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip(DATE_COL, title='Date/Time'),
            alt.Tooltip('Demand_MW', title='Demand (MW)', format=',.0f'),
            alt.Tooltip(TEMP_CELSIUS_COL, title='Temp (°C)', format='.1f'),
            alt.Tooltip('Detailed_Time_of_Day', title='Time of Day'),
            'Type'
        ]
    ).properties(
        title='Historical and Predicted Energy Demand Time-Series'
    )

    # 1. Historical Demand Line (Blue)
    historical_line = base.transform_filter(
        alt.datum.Type == ORIGINAL_TARGET_COL
    ).mark_line(color='steelblue').encode(
        # Group by 'Type' so lines don't connect across the gap between historical and predicted data
        detail='Type' 
    )

    # 2. Predicted Demand Line (Orange)
    predicted_line = base.transform_filter(
        alt.datum.Type == PREDICTION_COL_NAME
    ).mark_line(color='darkorange', strokeDash=[5,5]).encode(
        detail='Type'
    )
    
    # 3. Global Risk Threshold Line (Red Dashed)
    global_threshold_line = alt.Chart(pd.DataFrame({'y': [GLOBAL_RISK_THRESHOLD]})).mark_rule(color='red', strokeDash=[2,2]).encode(
        y='y',
        tooltip=[alt.Tooltip('y', title='Global Risk Threshold', format=',.0f')]
    )

    # 4. Dynamic Alert Line (Purple Dotted) - based on Moving Average
    dynamic_alert_line = alt.Chart(df_filtered).mark_line(color='darkviolet', strokeDash=[1,1]).encode(
        x=alt.X(DATE_COL),
        y=alt.Y('Dynamic_Alert'),
        tooltip=[alt.Tooltip('Dynamic_Alert', title='Dynamic Alert (MA + 500MW)', format=',.0f')]
    )
    
    # Combine the charts
    chart = (historical_line + predicted_line + dynamic_alert_line + global_threshold_line).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Calculate key metrics for display
    # Get the last date in the historical data to define the start of the forecast
    forecast_start_date = df_filtered[df_filtered[ORIGINAL_TARGET_COL].notna()][DATE_COL].max()
    
    # Get max predicted demand in the forecast period
    df_forecast = df_filtered[df_filtered[DATE_COL] > forecast_start_date]
    
    if not df_forecast.empty:
        peak_demand = df_forecast[PREDICTION_COL_NAME].max()
        peak_row = df_forecast.loc[df_forecast[PREDICTION_COL_NAME].idxmax()]
        peak_time_full = peak_row[DATE_COL].strftime('%Y-%m-%d %H:%M')
        peak_temp = peak_row[TEMP_CELSIUS_COL]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Forecast Peak Demand (MW)", 
                value=f"{peak_demand:,.2f}"
            )
        with col2:
            st.metric(
                label="Peak Time", 
                value=f"{peak_time_full}"
            )
        with col3:
            st.metric(
                label="Peak Temperature (°C)", 
                value=f"{peak_temp:.1f}°C"
            )
        
    st.markdown("---")
    st.markdown(f"**Global Risk Threshold (Red Dashed Line):** **{GLOBAL_RISK_THRESHOLD:,.0f} MW** (A historical high-water mark for potential shortage risk).")
    st.markdown(f"**Dynamic Alert Trigger (Purple Dotted Line):** **Moving Average ({MA_WINDOW}-hr) + {MA_ALERT_BUFFER} MW** (Flags a relative spike above normal operating levels).")

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Demand Forecast Dashboard")
    st.title("⚡ Simple Energy Demand Time-Series and Forecast")
    st.markdown(f"This dashboard visualizes historical and **predicted energy demand** over time, providing a simple view for your project on **early prediction of energy shortages** in Dutch neighborhoods.")

    df_full = load_data_and_predict()
    
    if df_full.empty:
        return

    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: TIME / FORECAST HORIZON ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Date & Horizon Selection")
    
    # Determine min/max dates for sliders
    min_date = df_full[DATE_COL].min().date()
    max_date = df_full[DATE_COL].max().date()
    
    # Use the last 7 days of the dataset as the default view
    default_start_date = max_date - timedelta(days=7)
    
    # Date Range Selector (The "Time" filter)
    selected_date_range = st.sidebar.slider(
        "Select Time Period (Date Range):",
        min_value=min_date,
        max_value=max_date,
        value=(default_start_date, max_date),
        format="YYYY-MM-DD",
        help="Adjust the slider to view a specific time window of historical and predicted data."
    )
    
    # Convert selected dates to datetime objects for filtering
    # Need to handle the timezone awareness of the DataFrame column
    start_dt = datetime.combine(selected_date_range[0], datetime.min.time(), tzinfo=df_full[DATE_COL].dt.tz).tz_convert('UTC')
    end_dt = datetime.combine(selected_date_range[1], datetime.max.time(), tzinfo=df_full[DATE_COL].dt.tz).tz_convert('UTC')

    # Forecast Horizon (Simple Toggle/Filter - implemented via the date slider)
    st.sidebar.info(
        "The **Forecast Horizon** is controlled by the **Time Period** slider."
        "The **Orange Dashed Line** represents the model's prediction, automatically extending from the last **Blue Line** data point."
    )
    
    # Filter the DataFrame based on the selected date range
    df_filtered = df_full[
        (df_full[DATE_COL] >= start_dt) & 
        (df_full[DATE_COL] <= end_dt)
    ].copy()

    # ----------------------------------------------------------------------
    # --- DASHBOARD PLOT (Demand vs Hour/Date) ---
    # ----------------------------------------------------------------------
    st.subheader(f"1. Energy Demand Over Time (Viewing {selected_date_range[0]} to {selected_date_range[1]})")
    
    create_demand_forecast_plot(df_filtered)

    st.markdown("---")
    st.markdown(f"This simple time-series plot directly addresses the need to monitor **demand over the hour/date** and identify potential shortage risks within the **forecast horizon** for your Dutch neighborhoods project.")


# Execute the main function
if __name__ == "__main__":
    main()
