import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta
import lightgbm as lgb
import altair as alt # Needed for combining charts (optional, but good practice)
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION (Must match training script) ---
DATE_COL = 'DateUTC'
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
MODEL_FILENAME = 'lightgbm_demand_model.joblib'

# List of categorical columns identified from the model's expected features
CATEGORICAL_COLS = ['MeasureItem', 'CountryCode', 'Time_of_Day', 'Detailed_Time_of_Day', 'CreateDate', 'UpdateDate']
TARGET_COL_SANITIZED = 'Demand_MW' # Sanitize result for TARGET_COL

# --- 1. UTILITY FUNCTIONS (Copied from Training Script) ---

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

# Feature Engineering Function - CRITICAL: Must mirror training features
def create_features(df):
    # FIX: Removed redundant column sanitization here. 
    # Columns must be sanitized only once in load_data to match model.feature_name_.
    
    # Ensure the required columns exist after sanitation
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    # Lag and Rolling Window Features (CRITICAL for recursive forecasting)
    
    # Lag 24 (The strongest behavioral feature)
    df['Demand_MW_lag24'] = df[TARGET_COL_SANITIZED].shift(24)
    
    # Lag 48 (The weekend/weekday cycle feature)
    df['Demand_MW_lag48'] = df[TARGET_COL_SANITIZED].shift(48)

    # 72-hour rolling mean, lagged by 24 hours (The stability feature)
    # This feature is calculated over the demand from t-96 to t-24
    df['Demand_MW_roll72'] = df[TARGET_COL_SANITIZED].shift(24).rolling(window=72).mean()

    # Temperature-related features (Assuming temperature is sanitized to a recognizable name)
    if TEMP_COL_SAN in df.columns:
        # Lag Temperature by 12 hours (since peak demand often follows temperature drop)
        df[TEMP_COL_SAN + '_lag12'] = df[TEMP_COL_SAN].shift(12)
    
    # Select the columns that were used for training (LightGBM handles the categorical features naturally)
    feature_cols = [col for col in df.columns if col != TARGET_COL_SANITIZED]
    
    # Return features. Note: X_t slicing with model.feature_name_ happens outside this function.
    return df[feature_cols]

# --- 2. CACHING AND LOADING ---

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the historical data."""
    try:
        df = pd.read_csv(file_path, parse_dates=[DATE_COL], index_col=DATE_COL)
        df.columns = sanitize_feature_names(df.columns)
        
        # Keep only the target and the essential weather feature
        required_cols = [TARGET_COL_SANITIZED, sanitize_feature_names([TEMP_COL])[0]]
        df = df[required_cols].dropna()

        # Resample to ensure hourly frequency for lag feature calculation
        df = df.resample('H').mean()
        
        # Only keep data that is recent enough for lags and rolling averages
        # Drop first few days where lags would be NaN
        df = df.iloc[96:] 

        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
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

# @st.cache_data(show_spinner="Running recursive forecast... This may take a moment.")
def run_recursive_forecast(historical_df, model, forecast_steps):
    """
    Performs a step-by-step recursive forecast.
    """
    st.info("Starting recursive prediction. Generating future weather inputs (simulated) and calculating features...")
    
    # 1. Setup Data
    last_known_time = historical_df.index[-1]
    
    # Create the index for the forecast (starting immediately after the last known time)
    forecast_index = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                   periods=forecast_steps, 
                                   freq='H')

    # Create the DataFrame to hold the forecast and simulated future features
    # NOTE: Future temp is simulated by repeating historical patterns.
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    future_temp_data = np.tile(historical_df[TEMP_COL_SAN].iloc[-96:].values, 
                               (forecast_steps // 96) + 1)[:forecast_steps]

    # Initialize the forecast DataFrame
    df_forecast = pd.DataFrame(index=forecast_index)
    df_forecast[TEMP_COL_SAN] = future_temp_data
    df_forecast[TARGET_COL_SANITIZED] = np.nan # This will hold our predictions

    # Combine historical and future data
    df_combined = pd.concat([historical_df, df_forecast])
    
    # 2. Perform Recursive Loop
    for t in forecast_index:
        
        # Create a temporary DataFrame for feature calculation, including the most recent actuals/predictions
        # We need enough historical/predicted data to calculate all lags and rolling means (96 hours total)
        df_temp = df_combined.loc[:t].copy() 
        
        # Use the most recent actuals/predictions to calculate features for time t
        # We need the last 48+72 hours to ensure roll72 and lag48 are calculated for the *last* row (time t)
        features_t = create_features(df_temp.tail(48 + 72)).tail(1)
        
        # Select the columns needed for the model
        # FIX: The model.feature_name_ contains the exact names the model expects.
        # We use this list to filter the features created by create_features.
        try:
            X_t = features_t[model.feature_name_]
        except KeyError as e:
            st.error(f"Model Feature Mismatch: A required feature ({e}) is missing or misnamed in the input data. Check feature engineering consistency.")
            return pd.DataFrame() # Stop the function on error

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

# --- 4. STREAMLIT APP LAYOUT ---

def display_daily_pattern(historical_df):
    """Allows user to select a date and views its 24-hour demand and temperature pattern."""
    st.subheader("2.1. Interactive Daily Pattern Viewer")
    st.markdown("Select a date to inspect the 24-hour energy demand (MW) versus temperature (°C) for that specific day, highlighting the evening ramp-up.")
    
    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]
    
    # Ensure index is datetime and extract dates
    dates = historical_df.index.normalize().unique()
    
    # Get the last date in the historical data as the default
    default_date = dates[-1].to_pydatetime().date()
    
    selected_date = st.date_input(
        "Select a Historical Date",
        value=default_date,
        min_value=dates.min().to_pydatetime().date(),
        max_value=dates.max().to_pydatetime().date()
    )

    # Filter data for the selected day
    selected_day_df = historical_df[historical_df.index.date == selected_date].copy()
    
    if selected_day_df.empty:
        st.warning(f"No data available for {selected_date}.")
        return

    # Prepare data for plotting
    selected_day_df['Hour'] = selected_day_df.index.hour
    selected_day_df['Demand_MW'] = selected_day_df[TARGET_COL_SANITIZED]
    # Convert Temperature (0.1 C) to Celsius for display
    selected_day_df['Temperature_C'] = selected_day_df[TEMP_COL_SAN] / 10.0
    
    # Altair Charts for dual axis plot
    
    # Chart 1: Demand (Left Axis)
    base = alt.Chart(selected_day_df).encode(
        x=alt.X('Hour:O', axis=alt.Axis(title='Hour of Day')),
    )

    demand_chart = base.mark_line(point=True, color='#006494').encode(
        y=alt.Y('Demand_MW:Q', axis=alt.Axis(title='Demand (MW)', titleColor='#006494')),
        tooltip=['Hour:O', alt.Tooltip('Demand_MW:Q', format=',.0f')]
    ).properties(
        title=f"Demand and Temperature on {selected_date.strftime('%Y-%m-%d')}"
    )

    # Chart 2: Temperature (Right Axis)
    temp_chart = base.mark_line(point=True, color='#E9573E').encode(
        y=alt.Y('Temperature_C:Q', axis=alt.Axis(title='Temperature (°C)', titleColor='#E9573E')),
        tooltip=['Hour:O', alt.Tooltip('Temperature_C:Q', format='.1f')]
    )
    
    # Combine charts using Altair's layering and resolve scales
    final_chart = alt.layer(demand_chart, temp_chart).resolve_scale(
        y='independent'
    ).interactive()
    
    st.altair_chart(final_chart, use_container_width=True)
    st.caption("Energy demand (blue) typically ramps up sharply in the evening, inversely correlated with temperature (orange).")
    st.markdown("---")

def display_historical_variance(historical_df):
    """Displays the historical demand by hour and temperature category."""
    st.subheader("2.2. Mean Demand by Hour and Temperature (Aggregate)")
    st.markdown("This aggregate chart confirms the strong relationship between **Time of Day** and **Temperature** over the entire historical period.")

    TEMP_COL_SAN = sanitize_feature_names([TEMP_COL])[0]

    # Function to categorize temperature (assuming temperature is in 0.1 degree Celsius, so divide by 10)
    def categorize_temp(temp):
        temp_c = temp / 10.0
        if temp_c < 5:
            return 'Cold (< 5°C)'
        elif temp_c < 15:
            return 'Mild (5°C - 15°C)'
        elif temp_c < 25:
            return 'Warm (15°C - 25°C)'
        else:
            return 'Hot (> 25°C)'

    # Create a copy of historical data to avoid modifying cached data
    df_analysis = historical_df.copy()

    # Apply categorization
    df_analysis['hour'] = df_analysis.index.hour
    
    if TEMP_COL_SAN in df_analysis.columns:
        df_analysis['Temp_Category'] = df_analysis[TEMP_COL_SAN].apply(categorize_temp)

        # Group by hour and temperature category, then calculate mean demand
        df_variance = df_analysis.groupby(['hour', 'Temp_Category'])[TARGET_COL_SANITIZED].mean().unstack()

        # Display the line chart
        if not df_variance.empty:
            st.line_chart(df_variance, use_container_width=True)
            st.caption("Each line represents the average demand across all historical data for the corresponding temperature band, showing variance over 24 hours.")
        else:
            st.warning("Historical analysis data is empty or missing required columns.")
    else:
        st.warning(f"Required column for variance analysis ({TEMP_COL_SAN}) is missing.")
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

    # --- 1, 2, 3. Find, Extract, and Display the Global Peak in the Q4 window ---
    
    if 'Predicted_Demand_MW' not in df_forecast.columns:
        st.error("Forecast column 'Predicted_Demand_MW' not found after filtering.")
        return

    # Find the row with the maximum predicted demand *within the Q4 window*
    peak_row_index = df_forecast['Predicted_Demand_MW'].idxmax()
    
    # Extract the peak hour, date, and value
    peak_demand = df_forecast.loc[peak_row_index, 'Predicted_Demand_MW']
    peak_hour = peak_row_index.strftime('%H:%M')
    peak_date = peak_row_index.strftime('%Y-%m-%d')
    
    st.subheader("Actionable Insight: Highest Predicted Peak Risk")

    col1, col2 = st.columns(2)
    
    with col1:
        # Highlight the peak value using st.metric
        st.metric(
            label="Critical Peak Demand Level",
            value=f"{peak_demand:,.0f} MW",
            delta=f"Forecasted Peak Time: {peak_hour}", 
            delta_color="off" # Remove the default green/red coloring
        )
    with col2:
        # Use a text block for the specific date
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #E9573E; border-left: 5px solid #E9573E; border-radius: 5px; height: 100px; background-color: #ffeaea;">
            <p style="margin-bottom: 5px; font-size: 14px; color: #555;">Worst-Case Date</p>
            <h3 style="margin: 0; color: #E9573E;">{peak_date}</h3>
            <p style="margin: 0; font-size: 12px; color: #999;">The moment requiring peak resource allocation in Q4.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- 4. Plot the Forecast with Annotation ---
    
    st.subheader(f"Q4 2025 Forecast ({DISPLAY_START_DATE.strftime('%b %d')} to Dec 31)")
    
    df_plot = df_forecast[['Predicted_Demand_MW']].copy()
    
    # Create a secondary column for highlighting the peak point
    df_plot['Peak_Highlight'] = df_plot['Predicted_Demand_MW']
    # Set all values except the peak to NaN, so only the peak point is plotted
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

    # 1. Historical Data Info
    st.sidebar.header("Data & Model Info")
    st.sidebar.success(f"Historical Data Loaded: {historical_df.shape[0]} records")
    st.sidebar.success(f"Model Loaded: {model.__class__.__name__}")
    st.sidebar.info(f"Last Actual Demand Date: {historical_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    
    # 2. Display EDA
    st.subheader("2. Exploratory Data Analysis (EDA)")
    
    # New feature: Daily Pattern Viewer
    display_daily_pattern(historical_df)

    # Existing feature: Aggregate Variance
    display_historical_variance(historical_df)


    # 3. Forecast Controls and Execution
    st.subheader("3. Generate Peak Forecast")
    
    # Define the forecast end date (Dec 31, 2025)
    FORECAST_END_DATE = datetime(2025, 12, 31, 23, 0, 0)
    last_known_time = historical_df.index[-1]

    # Calculate the required number of steps (hours)
    # The forecast must run from the end of the historical data to the end date
    forecast_range = pd.date_range(start=last_known_time + timedelta(hours=1), 
                                   end=FORECAST_END_DATE, 
                                   freq='H')
    FORECAST_HORIZON_HOURS = len(forecast_range)

    # Ensure the calculated horizon is positive and reasonable
    if FORECAST_HORIZON_HOURS <= 0:
        st.error("Historical data already covers the target period. Adjust FORECAST_END_DATE.")
        return

    st.info(f"The recursive model will run **{FORECAST_HORIZON_HOURS} hours** (starting after {last_known_time.strftime('%Y-%m-%d %H:%M')}) to accurately forecast the Q4 risk period.")

    if st.button(f'🚀 Run Forecast until Dec 31, 2025 ({FORECAST_HORIZON_HOURS} hours)', key='run_forecast_btn'):
        # Clear any old forecast data from the session state
        if 'df_forecast' in st.session_state:
            del st.session_state.df_forecast
        
        # Run the recursive function
        df_forecast = run_recursive_forecast(historical_df, model, FORECAST_HORIZON_HOURS)
        
        # Store the full result in session state
        st.session_state.df_forecast = df_forecast

    # 4. Display Forecast
    if 'df_forecast' in st.session_state and not st.session_state.df_forecast.empty:
        display_forecast_and_peak(st.session_state.df_forecast)
    elif st.button('Show Previous Forecast', key='show_forecast_btn', disabled='df_forecast' not in st.session_state):
        # Allow the user to view the previous forecast if it's stored
        if 'df_forecast' in st.session_state:
            display_forecast_and_peak(st.session_state.df_forecast)
    
if __name__ == '__main__':
    main()
