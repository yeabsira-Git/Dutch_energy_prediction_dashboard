import streamlit as st
import pandas as pd
import numpy as np
import altair as alt 
from datetime import datetime, date, timedelta
import warnings
import joblib
import lightgbm as lgb
import re
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATE_COL = 'DateUTC'
ORIGINAL_TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' 

GLOBAL_RISK_THRESHOLD = 15500 

# --- UTILITY FUNCTIONS ---

def get_temp_category(temp_c):
    """Categorizes temperature based on demand peak shifting findings."""
    if temp_c <= 10:
        return 'Cold (Peak Noon ≤ 10°C)'
    elif temp_c <= 20:
        return 'Mild (Peak Noon 10-20°C)'
    elif temp_c <= 25:
        return 'Warm (Peak Evening 20-25°C)'
    else:
        return 'Summer (Peak Evening > 25°C)'

def clean_column_names(df):
    """Standardizes column names to match the model's expected format (e.g., removing spaces/symbols)."""
    # Use re.sub to replace non-alphanumeric/underscore characters with an underscore
    def clean(col):
        # Replace non-alphanumeric characters (except underscore) with an underscore
        cleaned_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        # Remove leading/trailing underscores
        return cleaned_col.strip('_')
    
    df.columns = [clean(col) for col in df.columns]
    
    return df

# --- DATA LOADING AND PREDICTION (Highly Robust Function) ---

@st.cache_data
def load_data_and_predict():
    """
    Loads data, engineers features (including OHE), loads the model, and generates predictions.
    Includes robust data loading and feature checking to fix errors.
    """
    data_file_path = 'cleaned_energy_weather_data(1).csv' 
    model_file_path = MODEL_FILENAME
    
    # 1. Load Data with Robust Delimiter/Encoding Checks
    df = None
    try:
        # Attempt 1: Standard CSV (comma) with UTF-8
        df = pd.read_csv(
            data_file_path, parse_dates=[DATE_COL], encoding='utf-8', sep=',', skiprows=0, skip_blank_lines=True
        )
    except (pd.errors.ParserError, UnicodeDecodeError):
        try:
            # Attempt 2: European CSV (semicolon) with latin1
            df = pd.read_csv(
                data_file_path, parse_dates=[DATE_COL], encoding='latin1', sep=';', skiprows=0, skip_blank_lines=True
            )
        except Exception as e:
            st.error(f"FATAL DATA ERROR: Could not load data. Tried comma (UTF-8) and semicolon (Latin-1). (Error: {e})")
            st.stop()
    except Exception as e:
         st.error(f"FATAL DATA ERROR: Unexpected error during data load: {e}")
         st.stop()
         
    # 3a. Clean column names
    df = clean_column_names(df)
    
    # 2. Load Model
    try:
        model = joblib.load(model_file_path)
    except Exception as e:
        st.error(f"FATAL MODEL ERROR: Could not load model '{MODEL_FILENAME}'. Ensure file exists. (Error: {e})")
        st.stop()
    
    # Ensure DateUTC is timezone-aware
    if df[DATE_COL].dt.tz is None:
        df[DATE_COL] = df[DATE_COL].dt.tz_localize('UTC')
    
    # 3b. Feature Engineering
    # The actual column name from the CSV is 'Temperature_0_1_degrees_Celsius' after cleaning
    df[TEMP_CELSIUS_COL] = df['Temperature_0_1_degrees_Celsius'] / 10.0 

    # Temporal Features
    df['hour'] = df[DATE_COL].dt.hour
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df[DATE_COL].dt.month
    df['quarter'] = df[DATE_COL].dt.quarter
    df['dayofyear'] = df[DATE_COL].dt.dayofyear 
    df['weekofyear'] = df[DATE_COL].dt.isocalendar().week.astype(int) 
    df['time_index'] = df.index.to_series().astype(float) 

    # Lag and Rolling Features
    df['Demand_MW_lag24'] = df[ORIGINAL_TARGET_COL].shift(24)
    df['Demand_MW_lag48'] = df[ORIGINAL_TARGET_COL].shift(48)
    df['Demand_MW_roll72'] = df[ORIGINAL_TARGET_COL].shift(1).rolling(window=72).mean()
    df['temp_lag24'] = df[TEMP_CELSIUS_COL].shift(24)
    df['temp_roll72'] = df[TEMP_CELSIUS_COL].shift(1).rolling(window=72).mean()
    df['temp_roll168'] = df[TEMP_CELSIUS_COL].shift(1).rolling(window=168).mean()

    # 3c. Clean categorical values before OHE (CRITICAL FIX for OHE KeyErrors)
    categorical_cols = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate', 'Present_Weather_Code', 'Present_Weather_Code_Indicator']
    
    # Filter to only columns that exist and apply strip to values
    existing_categorical_cols = [c for c in categorical_cols if c in df.columns]

    for col in existing_categorical_cols:
        # Check if column is of object/string type and strip whitespace
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # Now perform OHE
    df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=False)
    
    # 3d. Final Feature Alignment (CRITICAL FIX): Ensure all expected OHE columns exist
    model_features = model.feature_name_
    
    # List of all OHE features from the error message
    expected_ohe_cols = [
        'MeasureItem_Monthly_Hourly_Load_Values', 'CountryCode_NL', 
        'CreateDate_03_03_2025_12_24_13', 'CreateDate_04_09_2025_15_45', 
        'UpdateDate_03_03_2025_12_24_13', 'UpdateDate_04_09_2025_15_45', 
        'Time_of_Day_Day', 'Time_of_Day_Night', 'Detailed_Time_of_Day_Evening', 
        'Detailed_Time_of_Day_Midnight', 'Detailed_Time_of_Day_Morning', 
        'Detailed_Time_of_Day_Night', 'Detailed_Time_of_Day_Noon',
        # Binary weather columns (cleaned names)
        'Fog_0_no_1_yes', 'Rainfall_0_no_1_yes', 'Snow_0_no_1_yes', 
        'Thunder_0_no_1_yes', 'Ice_Formation_0_no_1_yes'
    ]
    
    for col in expected_ohe_cols:
        if col not in df.columns:
            df[col] = 0 # Insert missing OHE column with a value of 0

    # Ensure numerical columns used by model have no NaNs (imputation)
    for feature in model_features:
        if feature in df.columns and df[feature].dtype in ['float64', 'int64']:
            df[feature] = df[feature].fillna(df[feature].mean())
    
    # 4. Filter for rows where prediction is possible
    # Drop rows where any of the required model features (including lags) are NaN
    df_predict = df.copy().dropna(subset=model_features, how='any')

    # 5. Predict (Strictly enforce feature order - FIX for LightGBMError)
    X_predict = df_predict[model_features] 
    y_pred = model.predict(X_predict)
    
    # 6. Merge prediction back
    df[PREDICTION_COL_NAME] = np.nan
    df.loc[df_predict.index, PREDICTION_COL_NAME] = y_pred
    
    # Add temperature category for visualization
    df['Weather_Category'] = df[TEMP_CELSIUS_COL].apply(get_temp_category)

    return df

# --- VISUALIZATION FUNCTIONS (Unchanged) ---

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
        # FIX: Explicitly set axis format to show both date and hour clearly
        x=alt.X(DATE_COL, title="Date and Time (UTC)", axis=alt.Axis(format="%Y-%m-%d %H:%M")),
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
    
    df_hourly = df.groupby(['hour', 'Weather_Category'])[PREDICTION_COL_NAME].mean().reset_index(name='Avg Predicted Demand (MW)')
    
    # Check if the dataframe is empty or if mode fails
    if df_hourly.empty:
        dominant_category = "N/A"
    else:
        # Safely determine the dominant category
        dominant_category_series = df['Weather_Category'].mode()
        if not dominant_category_series.empty:
            dominant_category = dominant_category_series.iloc[0].split('(')[1].replace(')', '').strip()
        else:
            dominant_category = "N/A"
    
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
    if df[PREDICTION_COL_NAME].empty or df[PREDICTION_COL_NAME].isnull().all():
        st.warning("No predictions available for the selected range.")
        return
        
    predicted_peak = df[PREDICTION_COL_NAME].max()
    predicted_low = df[PREDICTION_COL_NAME].min()
    
    peak_time_index = df[PREDICTION_COL_NAME].idxmax()
    low_time_index = df[PREDICTION_COL_NAME].idxmin()
    
    peak_time = df.loc[peak_time_index, DATE_COL].strftime('%Y-%m-%d %H:%M UTC')
    low_time = df.loc[low_time_index, DATE_COL].strftime('%Y-%m-%d %H:%M UTC')
    
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

# --- MAIN STREAMLIT APPLICATION (Unchanged) ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Shortage Prediction Prototype")
    st.title("💡 Energy Demand Prediction Prototype")
    # FIX: Updated title to reflect National Grid
    st.subheader(f"Project: Early Prediction of **National Grid** Energy Shortages in the Netherlands ({datetime.now().strftime('%Y-%m-%d')})")
    
    df_full = load_data_and_predict()
    
    # ----------------------------------------------------------------------
    # --- SIDEBAR CONTROLS (Slider in Calendar and Forecast Horizon Days) ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Controls & Settings")
    
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
    
    tz_info = df_full[DATE_COL].dt.tz
    start_dt = datetime.combine(selected_date_range[0], datetime.min.time()).replace(tzinfo=tz_info)
    end_dt = datetime.combine(selected_date_range[1], datetime.max.time()).replace(tzinfo=tz_info)

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
    
    # FEATURE 3: Demand vs Time of Day (Hourly Profile reflecting peak shift)
    st.subheader("4. Hourly Demand Profile: Visualizing Temperature-Based Peak Shift 🌡️")
    st.markdown("""
        This plot shows the **average predicted demand profile by hour (0-23)** for the selected period,
        highlighting the shift in peak demand (**Noon** for Cold/Mild vs. **Evening** for Warm/Summer).
    """)
    create_hourly_profile_plot(df_filtered)


# Execute the main application
if __name__ == "__main__":
    main()