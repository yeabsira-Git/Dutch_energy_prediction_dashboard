import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta
import lightgbm as lgb
import altair as alt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
MODEL_FILENAME = 'lightgbm_demand_model.joblib'

# List of categorical columns from the original dataset
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']
TARGET_COL_SANITIZED = 'Demand_MW' 

# --- 1. UTILITY FUNCTIONS (Feature Engineering & Sanitization) ---

def sanitize_feature_names(columns):
    """Helper function to sanitize column names for consistency."""
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

def create_features(df):
    """
    Creates all time-based, lag, and rolling window features expected by the model.
    This function must be consistent with the features generated during training.
    """
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    # 1. Time Features (Essential for both historical and forecast data)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    # Reconstructed feature from the error message
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int) 
    
    # 2. Demand Lag and Rolling Features
    if TARGET_COL_SANITIZED in df.columns:
        df['Demand_MW_lag24'] = df[TARGET_COL_SANITIZED].shift(24)
        df['Demand_MW_lag48'] = df[TARGET_COL_SANITIZED].shift(48)
        df['Demand_MW_roll72'] = df[TARGET_COL_SANITIZED].shift(24).rolling(window=72).mean()

    # 3. Temperature Lag and Rolling Features (Reconstructed from error list)
    if TEMP_COL_SAN in df.columns:
        df['temp_lag24'] = df[TEMP_COL_SAN].shift(24)
        df['temp_roll72'] = df[TEMP_COL_SAN].shift(24).rolling(window=72).mean()
        df['temp_roll168'] = df[TEMP_COL_SAN].shift(24).rolling(window=168).mean()
        
    # Return all generated columns (minus the target itself)
    feature_cols = [col for col in df.columns if col != TARGET_COL_SANITIZED]
    return df[feature_cols]

# --- 2. CACHING AND LOADING ---

@st.cache_data(show_spinner="Loading and aligning historical data...")
def load_data(file_path):
    """
    Loads, sanitizes all columns, applies one-hot encoding, and prepares historical data.
    This function must be consistent with the features generated during training.
    """
    try:
        # Load all data, setting the index and parsing dates
        df = pd.read_csv(file_path, parse_dates=[DATE_COL], index_col=DATE_COL)
        
        # Sanitize all column names
        df.columns = sanitize_feature_names(df.columns)
        
        # Ensure correct frequency and fill any missing hours with forward-fill (or first)
        df = df.resample('H').first()
        
        # Apply One-Hot Encoding for all original categorical columns
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dummy_na=False)

        # Drop rows where target is NaN (should be minimal after resample/first())
        df = df.dropna(subset=[TARGET_COL_SANITIZED])
        
        # Drop the first 168 hours to ensure all lag/roll features are non-NaN for the historical window
        df = df.iloc[168:] 

        return df
    except Exception as e:
        st.error(f"Error loading or processing data. Ensure 'cleaned_energy_weather_data(1).csv' is correctly formatted: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(file_path):
    """Loads the pre-trained LightGBM model."""
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. RECURSIVE FORECASTING LOGIC ---

def run_recursive_forecast(historical_df, model, forecast_steps):
    """Performs a step-by-step recursive forecast using simulated future weather."""
    st.info("Starting recursive prediction. Generating future weather inputs (simulated) and calculating features...")
    
    last_known_time = historical_df.index[-1]
    forecast_index = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                   periods=forecast_steps, 
                                   freq='H')

    # Get the static/weather columns from the historical data (all columns that are not the target)
    static_weather_cols = [col for col in historical_df.columns if col != TARGET_COL_SANITIZED]
    
    # 1. Setup future DataFrame by tiling recent historical weather data (168h cycle)
    df_forecast = pd.DataFrame(index=forecast_index)
    
    for col in static_weather_cols:
        # Replicate/Tile all original weather/static features for the forecast duration
        if col in historical_df.columns:
            historical_slice = historical_df[col].iloc[-168:]
            tiled_data = np.tile(historical_slice.values, (forecast_steps // 168) + 1)[:forecast_steps]
            df_forecast[col] = tiled_data
        else:
            # Handle OHE columns or other missing features by filling with 0
            df_forecast[col] = 0 
            
    # CRITICAL: Ensure OHE features for the future index are correctly set based on the date/time
    
    # Create necessary time-based categorical columns for the future index
    temp_future_df = pd.DataFrame(index=forecast_index)
    temp_future_df['Time_of_Day'] = temp_future_df.index.hour.map(lambda h: 'Day' if 6 <= h < 18 else 'Night')
    # Using a simplified hour mapping that is likely captured by the model features
    temp_future_df['Detailed_Time_of_Day'] = temp_future_df.index.hour.map({
        0: 'Midnight', 1: 'Midnight', 2: 'Midnight', 3: 'Midnight', 4: 'Midnight', 5: 'Midnight',
        6: 'Morning', 7: 'Morning', 8: 'Morning', 9: 'Morning', 10: 'Morning', 11: 'Morning',
        12: 'Noon', 13: 'Noon', 14: 'Noon', 15: 'Noon', 16: 'Noon', 17: 'Noon',
        18: 'Evening', 19: 'Evening', 20: 'Evening', 21: 'Evening', 22: 'Evening', 23: 'Evening'
    }) 

    # Perform OHE on the time columns
    future_ohe_df = pd.get_dummies(temp_future_df[['Time_of_Day', 'Detailed_Time_of_Day']], dummy_na=False)

    # Update df_forecast with the calculated time OHE features
    for col in future_ohe_df.columns:
        # Align column names for OHE features
        ohe_col_name = sanitize_feature_names([col])[0]
        if ohe_col_name in df_forecast.columns:
            df_forecast[ohe_col_name] = future_ohe_df[col].values
            
    df_forecast[TARGET_COL_SANITIZED] = np.nan # This will hold our predictions

    # Combine historical and future data
    df_combined = pd.concat([historical_df, df_forecast])
    
    # 2. Perform Recursive Loop
    for t in forecast_index:
        
        # Use enough data for lag/roll features (up to 168 hours needed for temp_roll168)
        df_temp = df_combined.loc[:t].copy() 
        # Calculate features on the trailing 168 hours, then take the last row
        features_t_raw = create_features(df_temp.tail(168)).tail(1)
        
        # CRITICAL: Align columns to model's exact expected features and fill any missing with 0
        try:
            # Use reindex to align the single row of generated features (features_t_raw) 
            # with the model's exact feature names, filling any missing OHE columns with 0.
            X_t = features_t_raw.reindex(columns=model.feature_name_, fill_value=0)
        except Exception as e:
            st.error(f"Failed to align features for prediction: {e}")
            return pd.DataFrame()

        # Predict the demand for time t
        pred_t = model.predict(X_t)[0]

        # Store the prediction (CRITICAL: used as input for the next step)
        df_combined.loc[t, TARGET_COL_SANITIZED] = pred_t

    # Final forecast is the predicted portion of the combined DataFrame
    final_forecast_df = df_combined.loc[forecast_index].rename(
        columns={TARGET_COL_SANITIZED: 'Predicted_Demand_MW'}
    )
    st.success("Forecast complete. Data is ready for analysis!")
    return final_forecast_df

# --- 4. STREAMLIT APP LAYOUT FUNCTIONS ---

def map_hour_to_detailed_time_of_day(hour):
    """Maps an hour (0-23) to a readable time segment."""
    if 0 <= hour < 6: return 'Night (00:00-05:59)'
    if 6 <= hour < 12: return 'Morning (06:00-11:59)'
    if 12 <= hour < 18: return 'Noon/Afternoon (12:00-17:59)'
    if 18 <= hour < 24: return 'Evening (18:00-23:59)'
    return 'Unknown'

def display_historical_daily_pattern(historical_df):
    """Allows user to select a date and views its 24-hour demand and temperature pattern."""
    st.subheader("2.1. Interactive Historical Daily Pattern Viewer")
    st.markdown("Select a historical date to inspect the 24-hour energy demand (MW) versus temperature (°C) for that specific day, highlighting the typical evening peak.")
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    # Ensure index is datetime and extract dates
    dates = historical_df.index.normalize().unique()
    
    default_date = dates[-1].to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Historical Date",
        value=default_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date(),
        key='historical_date_picker'
    )

    selected_day_df = historical_df[historical_df.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No data available for {selected_date}.")
        return

    # Prepare data for plotting
    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df[TARGET_COL_SANITIZED]
    
    # Convert 0.1 C to C for display
    if TEMP_COL_SAN in selected_day_df.columns:
        selected_day_df['Temperature_C'] = selected_day_df[TEMP_COL_SAN] / 10.0
    else:
        selected_day_df['Temperature_C'] = 0 # Fallback
    
    # Altair Charts for dual axis plot
    base = alt.Chart(selected_day_df).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day')),
    )

    demand_chart = base.mark_line(point=True, color='#006494').encode(
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Demand (MW)', titleColor='#006494')),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f')]
    ).properties(
        title=f"Demand and Temperature on {selected_date.strftime('%Y-%m-%d')}"
    )

    temp_chart = base.mark_line(point=True, color='#E9573E').encode(
        y=alt.Y('Temperature_C:Q', axis=alt.Axis(title='Temperature (°C)', titleColor='#E9573E')),
        tooltip=['Hour:O', alt.Tooltip('Temperature_C:Q', format='.1f')]
    )
    
    final_chart = alt.layer(demand_chart, temp_chart).resolve_scale(
        y='independent'
    ).interactive()
    
    st.altair_chart(final_chart, use_container_width=True)
    st.caption("Energy demand (blue) typically ramps up sharply in the evening, often inversely correlated with temperature (orange).")
    st.markdown("---")


def display_daily_peak_summary(selected_day_df, selected_date):
    """
    Analyzes the selected day's forecast and extracts the peak demand time and category.
    """
    st.subheader(f"5.1. Daily Peak Summary for {selected_date.strftime('%Y-%m-%d')}")
    
    peak_demand_row = selected_day_df['Predicted_Demand_MW'].idxmax()
    peak_demand = selected_day_df.loc[peak_demand_row, 'Predicted_Demand_MW']
    peak_hour = peak_demand_row.hour
    
    # Check the Time of Day Category (18-23 is Evening)
    peak_category = map_hour_to_detailed_time_of_day(peak_hour)
    
    # Time interval is hour:00 to hour+1:00
    peak_time_interval = f"{peak_hour:02d}:00 - {peak_hour+1:02d}:00"
    
    is_evening_peak = 'Evening' in peak_category
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Peak Demand (MW)", f"{peak_demand:,.0f}")
    
    with col2:
        st.metric("Peak Time Interval", peak_time_interval)
        
    with col3:
        # Custom display for the Evening Peak status
        status_style = "success" if is_evening_peak else "info"
        status_color = '#155724' if is_evening_peak else '#0c5460'
        bg_color = '#d4edda' if is_evening_peak else '#d1ecf1'
        border_color = '#c3e6cb' if is_evening_peak else '#bee5eb'
        
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; text-align: center; 
                    background-color: {bg_color};
                    border: 1px solid {border_color};">
            <p style="margin: 0; font-size: 14px; font-weight: 600;">Peak Time Category</p>
            <h4 style="margin: 5px 0 0; color: {status_color};">{peak_category.split('(')[0].strip()}</h4>
        </div>
        """, unsafe_allow_html=True)
        
    if is_evening_peak:
        st.success("**CONFIRMED:** The daily peak is driven by the high-risk Evening consumption window. This aligns with typical behavior patterns.")
    else:
        st.info("The daily peak occurred outside the typical Evening high-risk window, which may indicate a weather-driven or atypical demand pattern.")
    st.markdown("---")

def display_forecast_time_of_day_analysis(df_full_forecast):
    """
    Allows user to select a date and views its 24-hour predicted 
    demand pattern, highlighting the Time of Day effect.
    """
    st.subheader("5. Interactive Daily Pattern Viewer (Forecast)")
    st.markdown("Select any date from **July 1 to Dec 31, 2025** to inspect the predicted 24-hour demand pattern for that specific day.")
    
    # Filter the forecast data to the relevant Jul 1 - Dec 31 period
    DISPLAY_START_DATE = datetime(2025, 7, 1, 0, 0, 0)
    DISPLAY_END_DATE = datetime(2025, 12, 31, 23, 0, 0)
    
    # Filter data to the Jul-Dec range
    df_forecast = df_full_forecast.loc[DISPLAY_START_DATE:DISPLAY_END_DATE].copy()

    if df_forecast.empty:
        st.warning("Forecast data for Jul-Dec 2025 is not available. Please run the forecast first.")
        return
    
    dates = df_forecast.index.normalize().unique()
    
    if dates.empty:
        st.warning("No unique dates found in the Jul-Dec forecast range.")
        return

    # Default to the date with the highest peak in the forecast window
    peak_date = df_forecast['Predicted_Demand_MW'].idxmax().to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Forecast Date (Jul 1 - Dec 31, 2025)",
        value=peak_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date(),
        key='forecast_date_picker'
    )

    # Filter data for the selected day
    selected_day_df = df_forecast[df_forecast.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No forecast data available for {selected_date}.")
        return

    # --- Feature addition for visualization ---
    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df['Predicted_Demand_MW']
    
    # Add the categorical time of day column
    selected_day_df['Time_of_Day_Category'] = selected_day_df['Hour'].apply(map_hour_to_detailed_time_of_day)

    # --- Altair Plotting ---
    
    # Define order for categorical variable for clean legend/coloring
    category_order = ['Night (00:00-05:59)', 'Morning (06:00-11:59)', 'Noon/Afternoon (12:00-17:59)', 'Evening (18:00-23:59)']
    
    chart = alt.Chart(selected_day_df).mark_line(point=True).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day (0-23)')),
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Predicted Demand (MW)')),
        # Use color to show the time of day effect
        color=alt.Color('Time_of_Day_Category:N', sort=category_order, title="Time of Day"),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f'), 'Time_of_Day_Category:N']
    ).properties(
        title=f"Predicted Hourly Demand by Time Segment on {selected_date.strftime('%Y-%m-%d')}"
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.caption("The colored segments highlight how collective human behavior drives demand peaks.")
    
    # 5.1 Display the daily peak summary for the selected date
    display_daily_peak_summary(selected_day_df, selected_date)


def display_full_forecast_overview(df_full_forecast):
    """
    Filters the forecast to the Jul 1 - Dec 31 window and displays the overall peak risk.
    """
    st.header("📈 Full Demand Risk Forecast (Jul 1 - Dec 31, 2025)")

    # Define the period the user wants to see: Jul 1 to Dec 31
    DISPLAY_START_DATE = datetime(2025, 7, 1, 0, 0, 0)
    
    # Filter the forecast data to the relevant period
    df_forecast = df_full_forecast.loc[DISPLAY_START_DATE:].copy()
    
    if df_forecast.empty:
        st.warning(f"The generated forecast does not contain data starting from {DISPLAY_START_DATE.strftime('%Y-%m-%d')}. Rerun the forecast.")
        return

    # Find, Extract, and Display the Global Peak in the forecast window
    if 'Predicted_Demand_MW' not in df_forecast.columns:
        st.error("Forecast column 'Predicted_Demand_MW' not found after filtering.")
        return

    peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
    
    peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
    peak_hour_interval = f"{peak_row_index.hour:02d}:00 - {peak_row_index.hour+1:02d}:00"
    peak_date = peak_row_index.strftime('%Y-%m-%d')
    
    st.subheader("4. Actionable Insight: Highest Predicted Peak Risk")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Critical Peak Demand Level (Jul-Dec)",
            value=f"{peak_demand:,.0f} MW",
            delta=f"Peak Time Interval: {peak_hour_interval}", 
            delta_color="off" 
        )
    with col2:
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #E9573E; border-left: 5px solid #E9573E; border-radius: 5px; height: 100px; background-color: #ffeaea;">
            <p style="margin-bottom: 5px; font-size: 14px; color: #555;">Worst-Case Date (Jul-Dec)</p>
            <h3 style="margin: 0; color: #E9573E;">{peak_date}</h3>
            <p style="margin: 0; font-size: 12px; color: #999;'>The moment requiring peak resource allocation in the entire forecast period.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Plot the Forecast with Annotation
    st.subheader(f"4.1. Full Forecast Line Chart")
    
    df_plot = df_forecast[['Predicted_Demand_MW']].copy()
    
    df_plot['Peak_Highlight'] = df_plot['Predicted_Demand_MW']
    df_plot.loc[df_plot.index != peak_row_index, 'Peak_Highlight'] = pd.NA
    
    st.line_chart(df_plot, use_container_width=True, y=['Predicted_Demand_MW', 'Peak_Highlight'])
    
    st.caption("The recursive forecast for July 1 - Dec 31, 2025, with the single highest demand hour visually highlighted (Orange Dot).")

# --- MAIN EXECUTION ---

def main():
    st.set_page_config(layout="wide")
    st.title("💡 Predictive Energy Demand Risk Platform")
    st.markdown("#### LightGBM Recursive Forecast for Dutch Neighborhood Demand Peaks")

    # Load resources
    historical_df = load_data('cleaned_energy_weather_data(1).csv')
    model = load_model(MODEL_FILENAME)
    
    if historical_df.empty or model is None:
        return

    # 1. Sidebar Info
    st.sidebar.header("Data & Model Info")
    st.sidebar.success(f"Historical Data Loaded: {historical_df.shape[0]} records")
    st.sidebar.success(f"Model Loaded: {model.__class__.__name__}")
    st.sidebar.info(f"Last Actual Demand Date: {historical_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    # 2. Display Historical EDA
    st.subheader("2. Exploratory Data Analysis (EDA)")
    
    # Historical Daily Pattern Viewer (Kept as requested)
    display_historical_daily_pattern(historical_df)

    # 3. Forecast Controls and Execution
    st.subheader("3. Generate Peak Forecast")
    
    # NEW FORECAST END DATE: Dec 31, 2025
    FORECAST_END_DATE = datetime(2025, 12, 31, 23, 0, 0)
    last_known_time = historical_df.index[-1]

    # Calculate the required number of steps (hours)
    forecast_range = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                   end=FORECAST_END_DATE, 
                                   freq='H')
    FORECAST_HORIZON_HOURS = len(forecast_range)

    if FORECAST_HORIZON_HOURS <= 0:
        st.error("Historical data already covers the target period. Adjust FORECAST_END_DATE.")
        return

    st.info(f"The recursive model will run **{FORECAST_HORIZON_HOURS} hours** to accurately forecast the Jul 1 - Dec 31, 2025 period.")

    if st.button(f'🚀 Run Forecast until Dec 31, 2025', key='run_forecast_btn'):
        if 'df_forecast' in st.session_state:
            del st.session_state.df_forecast
        
        df_forecast = run_recursive_forecast(historical_df, model, FORECAST_HORIZON_HOURS)
        st.session_state.df_forecast = df_forecast

    # 4. Display Forecast
    if 'df_forecast' in st.session_state and not st.session_state.df_forecast.empty:
        # 4. & 4.1 - Peak insight and full Jul-Dec plot
        display_full_forecast_overview(st.session_state.df_forecast) 
        
        # 5. Interactive daily forecast viewer with time-of-day analysis
        display_forecast_time_of_day_analysis(st.session_state.df_forecast)
        
    elif st.button('Show Previous Forecast', key='show_forecast_btn', disabled='df_forecast' not in st.session_state):
        if 'df_forecast' in st.session_state:
            display_full_forecast_overview(st.session_state.df_forecast)
            display_forecast_time_of_day_analysis(st.session_state.df_forecast)
    
if __name__ == '__main__':
    main()
