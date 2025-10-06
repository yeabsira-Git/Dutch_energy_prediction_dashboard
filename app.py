import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, date
import lightgbm as lgb
import altair as alt 
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
MODEL_FILENAME = 'lightgbm_demand_model.joblib'
PREDICTION_COL_NAME = 'Predicted_Demand' 

CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']

# --- SCENARIO MAPPING (Used for fixed temperature input) ---
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

# --- PLOTTING FUNCTIONS ---

@st.cache_data
def create_historical_boxplot():
    """Generates and displays the historical demand distribution by temperature scenario using a box plot."""
    
    file_path = 'cleaned_energy_weather_data(1).csv' 
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Historical data file not found to generate context plot.")
        return

    # Preprocessing
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10
    
    # Mapping function (must be defined locally or imported)
    def map_temp_to_scenario(temp):
        """Maps continuous temperature (in Celsius) to one of the four scenarios."""
        if temp <= 10:
            return '1. Cold (≤ 10°C)'
        elif 10 < temp <= 20:
            return '2. Mild (10°C - 20°C)'
        elif 20 < temp <= 25:
            return '3. Warm (20°C - 25°C)'
        else: 
            return '4. Summer (> 25°C)'

    # Calculate and plot
    df['Scenario'] = df[TEMP_CELSIUS_COL].apply(map_temp_to_scenario)

    chart = alt.Chart(df).mark_boxplot(extent=1.5).encode(
        x=alt.X('Scenario:N', title='Temperature Scenario', sort=list(SCENARIO_MAP.keys())),
        y=alt.Y(TARGET_COL, title='Historical Demand (MW)', scale=alt.Scale(zero=False)),
        color=alt.Color('Scenario:N', title='Scenario'),
        tooltip=[
            'Scenario',
            alt.Tooltip(TARGET_COL, title='Demand (MW) Distribution', format=',.0f')
        ]
    ).properties(
        title='Historical Demand Distribution by Temperature Scenario'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown("_Box plots show the median (middle line), the interquartile range (box), and min/max range (whiskers) of historical demand for each scenario._")


def create_demand_plot(df_plot):
    """Plots only the predicted demand line (no risk thresholds)."""
    
    if df_plot.empty:
        st.info("No data available for the selected time range.")
        return
        
    base = alt.Chart(df_plot).encode( 
        x=alt.X(DATE_COL, title='Forecast Date (Hourly)'),
        tooltip=[
            DATE_COL, 
            alt.Tooltip(PREDICTION_COL_NAME, title="Demand (MW)", format=',.2f'),
            alt.Tooltip(TEMP_CELSIUS_COL, title="Scenario Temp (°C)"),
        ]
    )
    
    # Prediction Line (blue line)
    demand_line = base.mark_line(point=True).encode(
        y=alt.Y(PREDICTION_COL_NAME, title='Demand (MW)'),
        color=alt.value('darkblue'),
        strokeWidth=alt.value(2)
    )
    
    chart = demand_line.properties(
        title=f'Hourly Energy Demand Forecast for {df_plot[TEMP_CELSIUS_COL].iloc[0]:.1f}°C Scenario'
    )
    
    # Add interaction for the forecast plot
    st.altair_chart(chart.interactive(), use_container_width=True)


# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Demand Scenario Visualizer")
    st.title("☀️ Dutch Neighborhood Energy Demand Scenario Visualizer")
    st.markdown("A focused view on predicted demand for a fixed temperature scenario.")

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
    st.sidebar.header("Scenario and Time Controls")
    
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
    # --- DATA PREP FOR PLOTTING ---
    # ----------------------------------------------------------------------
    
    # Prepare the full 6-month prediction DataFrame
    df_full_plot = future_df.dropna(subset=[PREDICTION_COL_NAME])
    df_full_plot = df_full_plot.reset_index(names=[DATE_COL]) 
    
    # Filter the DataFrame based on the Date Calendar
    df_plot = df_full_plot[
        (df_full_plot[DATE_COL] >= start_date_filter) & 
        (df_full_plot[DATE_COL] <= end_date_filter) 
    ].copy()
    
    # Calculate simple peak metrics for summary
    peak_demand = df_plot[PREDICTION_COL_NAME].max()
    peak_row = df_plot.loc[df_plot[PREDICTION_COL_NAME].idxmax()]
    
    # ----------------------------------------------------------------------
    # --- DASHBOARD LAYOUT ---
    # ----------------------------------------------------------------------
    
    col_summary, col_boxplot = st.columns([1, 1])

    with col_summary:
        st.subheader("Forecast Summary")
        st.metric(
            label="Peak Predicted Demand (MW) in Selected Range", 
            value=f"{peak_demand:,.2f}", 
            delta=f"Scenario: {selected_scenario}",
            delta_color="off"
        )
        st.markdown(f"**Peak Time:** {peak_row[DATE_COL].strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"**Scenario Temperature:** **{temp_forecast_celsius:.1f}°C** (Fixed for Forecast)")
        st.markdown("---")
        st.markdown("Use the controls on the left to change the scenario and date range.")
        
    with col_boxplot:
        st.subheader("Historical Demand Context (Box Plot)")
        create_historical_boxplot()

    st.markdown("---")
    
    st.subheader("Hourly Energy Demand Forecast (Zoomable)")
    
    # Call the simplified plot function
    create_demand_plot(df_plot)


# Execute the main function
if __name__ == "__main__":
    main()