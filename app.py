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
    This must create a DataFrame with all columns the model was trained on.
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
    # This involves setting the Time_of_Day_... columns to 1 for the relevant index hour.
    
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

def display_historical_daily_pattern(historical_df):
    """Allows user to select a date and views its 24-hour demand and temperature pattern."""
    st.subheader("2. Interactive Historical Daily Pattern Viewer")
    st.markdown("Select a historical date to inspect the 24-hour energy demand (MW) versus temperature (°C) for that specific day, highlighting the typical evening peak.")
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    # Ensure index is datetime and extract dates
    dates = historical_df.index.normalize().unique()
    
    default_date = dates[-1].to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Historical Date",
        value=default_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date()
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

def map_hour_to_detailed_time_of_day(hour):
    """Maps an hour (0-23) to a readable time segment."""
    if 0 <= hour < 6: return 'Night (00:00-05:59)'
    if 6 <= hour < 12: return 'Morning (06:00-11:59)'
    if 12 <= hour < 18: return 'Noon/Afternoon (12:00-17:59)'
    if 18 <= hour < 24: return 'Evening (18:00-23:59)'
    return 'Unknown'

def display_forecast_time_of_day_analysis(df_full_forecast):
    """
    Allows user to select a date in Q4 2025 and views its 24-hour predicted 
    demand pattern, highlighting the Time of Day effect.
    """
    st.subheader("4.1. Interactive Forecast Day Analysis (Q4 Focus)")
    st.markdown("Select a date between **Oct 1 and Dec 31, 2025** to see the predicted 24-hour demand pattern, colored by the time of day. This visually confirms the **Evening Peak** is the highest risk period.")
    
    # Filter the forecast data to the relevant Q4 period
    DISPLAY_START_DATE = datetime(2025, 10, 1, 0, 0, 0)
    DISPLAY_END_DATE = datetime(2025, 12, 31, 23, 0, 0)
    
    # Filter data to the Q4 range
    df_forecast = df_full_forecast.loc[DISPLAY_START_DATE:DISPLAY_END_DATE].copy()

    if df_forecast.empty:
        st.warning("Forecast data for Q4 2025 is not available. Please run the forecast first.")
        return
    
    dates = df_forecast.index.normalize().unique()
    
    if dates.empty:
        st.warning("No unique dates found in the Q4 forecast range.")
        return

    # Default to the date with the highest peak in the Q4 forecast
    peak_date = df_forecast['Predicted_Demand_MW'].idxmax().to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Forecast Date (Oct 1 - Dec 31, 2025)",
        value=peak_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date()
    )

    # Filter data for the selected day
    selected_day_df = df_forecast[df_forecast.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No forecast data available for {selected_date} in the Q4 window.")
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
    st.caption("The colored segments highlight how collective human behavior (like returning home and cooking in the **Evening**) drives the highest demand peaks.")
    st.markdown("---")


def display_forecast_and_peak(df_full_forecast):
    """
    Filters the forecast to the Oct 1 - Dec 31 window and displays the peak risk.
    """
    st.header("📈 Q4 2025 Energy Demand Risk Forecast")

    # Define the period the user wants to see: Oct 1 to Dec 31
    DISPLAY_START_DATE = datetime(2025, 10, 1, 0, 0, 0)
    
    # Filter the forecast data to the relevant Q4 period
    df_forecast = df_full_forecast.loc[DISPLAY_START_DATE:].copy()
    
    if df_forecast.empty:
        st.warning(f"The generated forecast does not contain data starting from {DISPLAY_START_DATE.strftime('%Y-%m-%d')}. Rerun the forecast.")
        return

    # Find, Extract, and Display the Global Peak in the Q4 window
    if 'Predicted_Demand_MW' not in df_forecast.columns:
        st.error("Forecast column 'Predicted_Demand_MW' not found after filtering.")
        return

    peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
    
    peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
    peak_hour = peak_row_index.strftime('%H:%M')
    peak_date = peak_row_index.strftime('%Y-%m-%d')
    
    st.subheader("3. Actionable Insight: Highest Predicted Peak Risk")

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Critical Peak Demand Level",
            value=f"{peak_demand:,.0f} MW",
            delta=f"Forecasted Peak Time: {peak_hour}", 
            delta_color="off" 
        )
    with col2:
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #E9573E; border-left: 5px solid #E9573E; border-radius: 5px; height: 100px; background-color: #ffeaea;">
            <p style="margin-bottom: 5px; font-size: 14px; color: #555;">Worst-Case Date</p>
            <h3 style="margin: 0; color: #E9573E;">{peak_date}</h3>
            <p style="margin: 0; font-size: 12px; color: #999;">The moment requiring peak resource allocation in Q4.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Plot the Forecast with Annotation
    st.subheader(f"4. Q4 2025 Forecast Overview ({DISPLAY_START_DATE.strftime('%b %d')} to Dec 31)")
    
    df_plot = df_forecast[['Predicted_Demand_MW']].copy()
    
    df_plot['Peak_Highlight'] = df_plot['Predicted_Demand_MW']
    df_plot.loc[df_plot.index != peak_row_index, 'Peak_Highlight'] = pd.NA
    
    st.line_chart(df_plot, use_container_width=True, y=['Predicted_Demand_MW', 'Peak_Highlight'])
    
    st.caption("The recursive forecast for Q4 2025, with the single highest demand hour visually highlighted (Orange Dot).")

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

    st.info(f"The recursive model will run **{FORECAST_HORIZON_HOURS} hours** to accurately forecast the Q4 risk period (Oct 1 - Dec 31, 2025).")

    if st.button(f'🚀 Run Forecast until Dec 31, 2025', key='run_forecast_btn'):
        if 'df_forecast' in st.session_state:
            del st.session_state.df_forecast
        
        df_forecast = run_recursive_forecast(historical_df, model, FORECAST_HORIZON_HOURS)
        st.session_state.df_forecast = df_forecast

    # 4. Display Forecast
    if 'df_forecast' in st.session_state and not st.session_state.df_forecast.empty:
        # 3.1 & 3.2 - Peak insight and full Q4 plot
        display_forecast_and_peak(st.session_state.df_forecast) 
        
        # 4.1 - New interactive daily forecast viewer with time-of-day analysis
        display_forecast_time_of_day_analysis(st.session_state.df_forecast)
        
    elif st.button('Show Previous Forecast', key='show_forecast_btn', disabled='df_forecast' not in st.session_state):
        if 'df_forecast' in st.session_state:
            display_forecast_and_peak(st.session_state.df_forecast)
            display_forecast_time_of_day_analysis(st.session_state.df_forecast)
    
if __name__ == '__main__':
    main()
