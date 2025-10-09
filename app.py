import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta, time
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
DATA_FILENAME = 'cleaned_energy_weather_data(1).csv'

# List of categorical columns from the original dataset
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'CreateDate', 'UpdateDate', 'Value_ScaleTo100', 'DateShort', 'TimeFrom', 'TimeTo', 'index']

TARGET_COL_SANITIZED = 'Demand_MW' 

# Define prediction limits
FORECAST_START_DATE_LIMIT = datetime(2025, 7, 1).date() 
FORECAST_END_DATE_LIMIT = datetime(2025, 12, 31).date() 

# --- 1. CACHING FUNCTIONS (To solve the 2-minute delay) ---

@st.cache_data
def load_and_prepare_data(filepath, date_col):
    """
    Loads and preprocesses data, caching the result to run only once.
    FIXED: The function now correctly keeps rows where Demand_MW is NaN (future forecast data).
    """
    st.info("⏳ Initial data loading and processing... This happens only once.")
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # --- CRITICAL FIX: Removed TARGET_COL from dropna subset ---
        # We only drop rows missing critical INPUT features (Temperature, Time_of_Day) 
        # but keep rows where the output (Demand_MW) is missing (i.e., the forecast data).
        df = df.dropna(subset=[TEMP_COL, 'Time_of_Day']).drop_duplicates()
        # -----------------------------------------------------------

        # --- Column Selection and Indexing Fix ---
        final_cols = [TARGET_COL]
        all_feature_cols = [col for col in df.columns if col not in CATEGORICAL_COLS]
        final_cols.extend(all_feature_cols)
        final_cols = list(pd.unique(final_cols))
        
        if date_col in final_cols:
            final_cols.remove(date_col) 
            
        df = df[[date_col] + final_cols].set_index(date_col) 
        
        # --- Index Mismatch Fix ---
        df.index = df.index.tz_localize(None) 
        df = df.sort_index()
        df = df.asfreq('H') 
        # ---------------------------
        
        st.success("Data loaded and ready!")
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file {filepath} not found. Ensure it is in the app directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(filepath):
    """Loads the pre-trained LightGBM model, caching the result to run only once."""
    st.info("⏳ Loading the LightGBM model... This happens only once.")
    try:
        model = joblib.load(filepath)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file {filepath} not found. Ensure it is in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. UTILITY FUNCTIONS (Feature Engineering & Sanitization) ---

def sanitize_feature_names(columns):
    """Helper function to sanitize column names for consistency."""
    new_cols = []
    for col in columns:
        col = str(col)
        col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        col = re.sub(r'^_+|_+$', '', col)
        col = re.sub(r'_+', '_', col)
        new_cols.append(col)
    return new_cols

def create_time_features(df):
    """Creates time-based features required by the LightGBM model."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def map_hour_to_time_of_day(hour):
    """Maps hour to the Time_of_Day category used in the training data."""
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 24: return 'Evening'
    else: return 'Night'

def map_hour_to_detailed_time_of_day(hour):
    """Maps hour to the Detailed_Time_of_Day category (critical for your findings)."""
    if 5 <= hour < 9: return 'Early Morning'
    elif 9 <= hour < 12: return 'Late Morning'
    elif 12 <= hour < 15: return 'Early Afternoon'
    elif 15 <= hour < 17: return 'Late Afternoon'
    elif 17 <= hour < 20: return 'Early Evening' 
    elif 20 <= hour < 24: return 'Late Evening'
    else: return 'Night'

# --- 3. PREDICTION FUNCTION ---

def predict_24h_demand(target_date, model, df_data, target_col):
    """Generates features and runs the 24-hour prediction."""
    try:
        start_dt = datetime.combine(target_date, time(0, 0))
        end_dt = start_dt + timedelta(days=1)
        
        # FIX: Using inclusive='left' for compatibility and 24 hours
        future_index = pd.date_range(start=start_dt, end=end_dt, freq='H', inclusive='left')
        
        # Retrieve the 24 hours of data from the cached data frame
        df_target = df_data.loc[future_index].copy()
        
        df_target = create_time_features(df_target)
        
        df_target['Time_of_Day'] = df_target.index.hour.map(map_hour_to_time_of_day)
        df_target['Detailed_Time_of_Day'] = df_target.index.hour.map(map_hour_to_detailed_time_of_day)
        
        df_target.columns = sanitize_feature_names(df_target.columns)
        
        model_features = model.feature_name() 
        X_target = df_target[model_features]
        
        predictions = model.predict(X_target)
        
        df_result = pd.DataFrame(predictions, index=future_index, columns=[f'Predicted_{target_col}'])
        
        df_result['Temperature'] = df_target['Temperature_0_1_degrees_Celsius'] / 10 
        df_result['Time_of_Day'] = df_target['Time_of_Day']
        df_result['Detailed_Time_of_Day'] = df_target['Detailed_Time_of_Day']
        
        return df_result
    
    except KeyError as e:
        st.error(f"Missing required data for prediction on {target_date.strftime('%Y-%m-%d')}. Check if the date is in your {DATA_FILENAME} forecast range. Missing key: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# --- 4. DISPLAY FUNCTIONS ---

def display_daily_forecast_chart(df_forecast, target_date, df_data):
    """Renders the Altair chart including historical data."""
    
    # 1. Define Historical Period (7 days prior to forecast start)
    forecast_start = datetime.combine(target_date, time(0, 0))
    history_start = forecast_start - timedelta(days=7)
    
    # 2. Extract Historical Data (Actual Demand)
    # Filter for target column and rename
    df_history = df_data.loc[history_start:forecast_start - timedelta(hours=1), [TARGET_COL]].copy()
    df_history = df_history.rename(columns={TARGET_COL: 'Demand_MW'}).reset_index()
    df_history['Type'] = 'Actual Demand'

    # 3. Prepare Forecast Data (Predicted Demand)
    df_forecast_plot = df_forecast[['Predicted_Demand_MW']].copy()
    df_forecast_plot = df_forecast_plot.rename(columns={'Predicted_Demand_MW': 'Demand_MW'}).reset_index()
    df_forecast_plot['Type'] = 'Predicted Demand'
    
    # 4. Combine and Rename Columns for Plotting
    df_plot = pd.concat([df_history, df_forecast_plot])
    df_plot = df_plot.rename(columns={'index': 'Hour'})
    
    # Identify the predicted peak for annotation
    peak_demand = df_forecast_plot['Demand_MW'].max()
    
    # Create the base chart
    base = alt.Chart(df_plot).encode(
        x=alt.X('Hour', title='Date and Time', axis=alt.Axis(format='%Y-%m-%d %H:%M')),
        y=alt.Y('Demand_MW', title='Demand (MW)'),
        color=alt.Color('Type', legend=alt.Legend(title="Demand Type"), scale=alt.Scale(range=['#2c7bb6', '#d7191c'])), # Blue for actual, Red for predicted
        tooltip=['Hour', alt.Tooltip('Demand_MW', format=',.0f'), 'Type']
    ).properties(
        title=f"Energy Demand: Last 7 Days (Actual) vs. {target_date.strftime('%Y-%m-%d')} (Predicted)"
    )

    # Line chart for demand
    line = base.mark_line().encode(
        strokeDash=alt.condition(
            alt.datum.Type == 'Predicted Demand',
            alt.value([5, 5]),  # Dashed line for prediction
            alt.value([1, 0])   # Solid line for actual
        )
    )
    
    # Add a vertical line to mark the start of the prediction
    vertical_line = alt.Chart(pd.DataFrame({'x': [forecast_start]})).mark_rule(color='gray', strokeWidth=2).encode(
        x='x:T',
        size=alt.value(2)
    )

    # Text label for the predicted peak
    peak_label = alt.Chart(df_forecast_plot).mark_text(
        align='left',
        baseline='middle',
        dx=5, 
        dy=-10, 
        color='red'
    ).encode(
        x='Hour',
        y='Demand_MW',
        text=alt.condition(
            alt.datum.Demand_MW == peak_demand,
            alt.Text('Demand_MW', format=',.0f'),
            alt.value('')
        )
    )

    st.altair_chart(line + vertical_line + peak_label, use_container_width=True)

def display_daily_peak_summary(df_forecast):
    """Displays the key summary findings, focusing on the Dinner Hour peak."""
    peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
    peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
    peak_time_interval = f"{peak_row_index.hour:02d}:00 - {peak_row_index.hour+1:02d}:00"
    peak_category = df_forecast.loc[peak_row_index, 'Detailed_Time_of_Day']

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Peak Demand", value=f"{peak_demand:,.0f} MW", delta_color="inverse")
    
    with col2:
        st.metric(label="Predicted Peak Hour", value=peak_time_interval)

    with col3:
        if 'Evening' in peak_category:
            st.markdown(f"<p style='color:red; font-size:18px;'>⚠️ **Dinner Hour Peak**</p>", unsafe_allow_html=True)
            st.metric(label="Time Category", value=peak_category)
        else:
            st.metric(label="Time Category", value=peak_category)
    
    st.markdown(f"""
    <p style='font-size: 14px; margin-top: 10px;'>
    The model predicts the highest demand occurs during the <strong>{peak_category}</strong>,
    confirming the dominant influence of collective human behavior (like the Dutch dinner hour)
    on high demand peaks.
    </p>
    """, unsafe_allow_html=True)


# --- 5. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Demand Peak Prediction")

    # --- 5.1 Load Cached Resources ---
    df_data = load_and_prepare_data(DATA_FILENAME, DATE_COL)
    model = load_model(MODEL_FILENAME)
    
    if df_data.empty or model is None:
        st.warning("Cannot run prediction without data or model. Please ensure files are correctly named and located.")
        return

    # --- 5.2 Header and Introduction ---
    st.title("⚡ Early Prediction of High Demand Peaks (Netherlands)")
    st.markdown("""
    This prototype uses a LightGBM model to predict 24-hour energy demand, highlighting the predictable **Dinner Hour** peaks.
    """)
    st.markdown("---")
    
    # --- 5.3 Prediction Input ---
    st.subheader("1. Select Target Date")
    
    DEMO_DATE_DT = datetime(2025, 10, 12).date() 

    target_date = st.date_input(
        "Select Target Date for 24-Hour Forecast:",
        min_value=FORECAST_START_DATE_LIMIT,
        max_value=FORECAST_END_DATE_LIMIT,
        value=DEMO_DATE_DT 
    )

    # --- 5.4 Prediction Execution ---
    st.subheader("2. Run 24-Hour Forecast")

    if st.button("Generate Forecast", type="primary"):
        target_date_dt = target_date.date() if isinstance(target_date, datetime) else target_date
        
        if target_date_dt < FORECAST_START_DATE_LIMIT or target_date_dt > FORECAST_END_DATE_LIMIT:
            st.error(f"Selected date is outside the valid forecast range ({FORECAST_START_DATE_LIMIT} to {FORECAST_END_DATE_LIMIT}).")
        else:
            with st.spinner(f"Predicting demand for {target_date_dt.strftime('%Y-%m-%d')}..."):
                df_forecast = predict_24h_demand(target_date_dt, model, df_data, TARGET_COL_SANITIZED)

            if df_forecast is not None and not df_forecast.empty:
                st.session_state.daily_forecast = df_forecast
                st.session_state.target_date = target_date_dt 
                
                peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
                peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
                peak_time_interval = f"{peak_row_index.hour:02d}:00 - {peak_row_index.hour+1:02d}:00"
                peak_category = map_hour_to_detailed_time_of_day(peak_row_index.hour)
                
                if 'Evening' in peak_category:
                    st.toast(f"🚨 PEAK ALERT! Predicted peak of {peak_demand:,.0f} MW at {peak_time_interval} (Evening) on {target_date_dt.strftime('%Y-%m-%d')}!", icon='🔥')
                else:
                    st.toast(f"✅ Prediction complete. Peak of {peak_demand:,.0f} MW at {peak_time_interval} on {target_date_dt.strftime('%Y-%m-%d')}.", icon='💡')

    # --- 5.5 Display Daily Forecast Results ---
    st.subheader("3. Forecast Results")
    
    if 'daily_forecast' in st.session_state and not st.session_state.daily_forecast.empty:
        
        # Display the 24-hour chart, including historical data
        display_daily_forecast_chart(st.session_state.daily_forecast, st.session_state.target_date, df_data)
        
        # Display the peak summary and risk warning
        display_daily_peak_summary(st.session_state.daily_forecast)
        
    else:
        st.info("Click 'Generate Forecast' above to see the 24-hour prediction.")


if __name__ == "__main__":
    if 'daily_forecast' not in st.session_state:
        st.session_state.daily_forecast = pd.DataFrame()
    if 'target_date' not in st.session_state:
        st.session_state.target_date = datetime(2025, 10, 12).date()
        
    main()