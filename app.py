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

# The 'Time_of_Day' columns are features needed for prediction, so they are not in CATEGORICAL_COLS 
# (which are typically dropped, but we need these specific ones).
TARGET_COL_SANITIZED = 'Demand_MW' 

# Define prediction limits
FORECAST_START_DATE_LIMIT = datetime(2025, 7, 1).date() 
FORECAST_END_DATE_LIMIT = datetime(2025, 12, 31).date() 

# --- 1. CACHING FUNCTIONS (To solve the 2-minute delay) ---

# Use st.cache_data for functions that return data frames/objects
@st.cache_data
def load_and_prepare_data(filepath, date_col):
    """
    Loads and preprocesses data, caching the result to run only once.
    This resolves the slowness caused by reading the large CSV repeatedly.
    """
    st.info("⏳ Initial data loading and processing... This happens only once.")
    try:
        # Load the data (The slow step)
        df = pd.read_csv(filepath)
        
        # Ensure date column is datetime and drop duplicates/NaNs
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Drop rows where critical columns are missing
        df = df.dropna(subset=[TARGET_COL, TEMP_COL, 'Time_of_Day']).drop_duplicates()

        # --- FIX: Ensure unique and correct column selection for set_index ---
        # 1. Start with the target column
        final_cols = [TARGET_COL]
        
        # 2. Add all non-categorical/metadata columns (including weather features)
        all_feature_cols = [col for col in df.columns if col not in CATEGORICAL_COLS]
        final_cols.extend(all_feature_cols)
        
        # 3. Ensure the list contains only unique column names
        final_cols = list(pd.unique(final_cols))
        
        # 4. Remove the date column from the feature list, as it's used for the index
        if date_col in final_cols:
            final_cols.remove(date_col) 
            
        # 5. FINAL SELECTION: Use the unique list and set the datetime column as the index
        # This resolves the ValueError
        df = df[[date_col] + final_cols].set_index(date_col) 
        # ----------------------------------------------------------------------
        
        st.success("Data loaded and ready!")
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file {filepath} not found. Ensure it is in the app directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

# Use st.cache_resource for objects that should persist, like ML models
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
    
    # Example for sine/cosine features (often used for time cyclicity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def map_hour_to_time_of_day(hour):
    """Maps hour to the Time_of_Day category used in the training data."""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

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
        # Create a 24-hour timestamp index for the target day
        start_dt = datetime.combine(target_date, time(0, 0))
        end_dt = start_dt + timedelta(days=1)
        # Use closed='left' for hourly data
        future_index = pd.date_range(start=start_dt, end=end_dt, freq='H', closed='left')
        
        # 1. Extract known weather data for the target day from the full dataset
        # .loc[] is used here to quickly retrieve the data needed for the target date
        df_target = df_data.loc[future_index].copy()
        
        # 2. Add Time Features
        df_target = create_time_features(df_target)
        
        # 3. Add Time_of_Day features (Critical for your model)
        df_target['Time_of_Day'] = df_target.index.hour.map(map_hour_to_time_of_day)
        df_target['Detailed_Time_of_Day'] = df_target.index.hour.map(map_hour_to_detailed_time_of_day)
        
        # 4. Final Feature Sanitization (must match training features)
        df_target.columns = sanitize_feature_names(df_target.columns)
        
        # 5. Select features used in the trained model
        model_features = model.feature_name() 
        X_target = df_target[model_features]
        
        # 6. Predict
        predictions = model.predict(X_target)
        
        df_result = pd.DataFrame(predictions, index=future_index, columns=[f'Predicted_{target_col}'])
        
        # Merge key input columns back for visualization/context
        # Note: Temperature column name is sanitized in df_target
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

def display_daily_forecast_chart(df_forecast, target_date):
    """Renders the Altair chart for the 24-hour forecast."""
    df_plot = df_forecast.reset_index().rename(columns={'index': 'Hour'})
    
    # Identify the predicted peak
    peak_demand = df_plot['Predicted_Demand_MW'].max()
    
    # Create the base chart
    base = alt.Chart(df_plot).encode(
        x=alt.X('Hour', title='Time of Day', axis=alt.Axis(format='%H:%M'))
    ).properties(
        title=f"24-Hour Energy Demand Forecast: {target_date.strftime('%Y-%m-%d')}"
    )

    # Line chart for demand
    line = base.mark_line(point=True).encode(
        y=alt.Y('Predicted_Demand_MW', title='Demand (MW)'),
        color=alt.value("#005C99"), 
        tooltip=['Hour', alt.Tooltip('Predicted_Demand_MW', format=',.0f'), 'Detailed_Time_of_Day', alt.Tooltip('Temperature', format='.1f')]
    )
    
    # Point for the peak demand
    peak_point = base.mark_circle(size=80, color='red').encode(
        y='Predicted_Demand_MW',
        opacity=alt.condition(
            alt.datum.Predicted_Demand_MW == peak_demand,
            alt.value(1),
            alt.value(0)
        )
    )
    
    # Text label for the peak
    peak_label = base.mark_text(
        align='left',
        baseline='middle',
        dx=5, # move text right
        dy=-10, # move text up
        color='red'
    ).encode(
        y='Predicted_Demand_MW',
        text=alt.condition(
            alt.datum.Predicted_Demand_MW == peak_demand,
            alt.Text('Predicted_Demand_MW', format=',.0f'),
            alt.value('')
        )
    )

    st.altair_chart(line + peak_point + peak_label, use_container_width=True)

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
        # Highlight the 'Evening' based on your presentation's key finding
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
    # These functions run ONCE and will be fast on subsequent runs.
    # The caching resolves the 2-minute delay.
    df_data = load_and_prepare_data(DATA_FILENAME, DATE_COL)
    model = load_model(MODEL_FILENAME)
    
    # Check if essential resources are loaded
    if df_data.empty or model is None:
        st.warning("Cannot run prediction without data or model. Please ensure files are correctly named and located.")
        return

    # --- 5.2 Header and Introduction ---
    st.title("⚡ Early Prediction of High Demand Peaks (Netherlands)")
    st.markdown("""
    This prototype uses a LightGBM model to predict 24-hour energy demand based on weather forecasts and time-based features.
    The goal is to accurately forecast the **Dinner Hour** peaks caused by collective human behavior.
    """)
    st.markdown("---")
    
    # --- 5.3 Prediction Input ---
    st.subheader("1. Select Target Date")
    
    # Set the default date to October 12, 2025, for the demo
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
        # Convert to date object for comparison
        target_date_dt = target_date.date() if isinstance(target_date, datetime) else target_date
        
        # Check if the date is in the forecast range
        if target_date_dt < FORECAST_START_DATE_LIMIT or target_date_dt > FORECAST_END_DATE_LIMIT:
            st.error(f"Selected date is outside the valid forecast range ({FORECAST_START_DATE_LIMIT} to {FORECAST_END_DATE_LIMIT}).")
        else:
            # Run the prediction (now fast due to caching)
            with st.spinner(f"Predicting demand for {target_date_dt.strftime('%Y-%m-%d')}..."):
                df_forecast = predict_24h_demand(target_date_dt, model, df_data, TARGET_COL_SANITIZED)

            if df_forecast is not None and not df_forecast.empty:
                st.session_state.daily_forecast = df_forecast
                
                # Instant Peak Alert
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
        
        # Display the 24-hour chart
        display_daily_forecast_chart(st.session_state.daily_forecast, target_date)
        
        # Display the peak summary and risk warning
        display_daily_peak_summary(st.session_state.daily_forecast)
        
    else:
        st.info("Click 'Generate Forecast' above to see the 24-hour prediction.")


if __name__ == "__main__":
    # Ensure session state is initialized
    if 'daily_forecast' not in st.session_state:
        st.session_state.daily_forecast = pd.DataFrame()
        
    main()