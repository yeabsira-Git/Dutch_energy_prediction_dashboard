import streamlit as st
import pandas as pd
import numpy as np
import altair as alt 
from datetime import datetime
import warnings
import joblib
import lightgbm as lgb

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATE_COL = 'DateUTC'
ORIGINAL_TARGET_COL = 'Demand_MW'
VIS_TARGET_COL = 'Predicted_Demand' 
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
MODEL_FILENAME = 'lightgbm_demand_model.joblib'

# --- SCENARIO MAPPING ---
SCENARIO_MAP = {
    "1. Cold (≤ 10°C)": 5.0,     
    "2. Mild (10°C - 20°C)": 15.0,   
    "3. Warm (20°C - 25°C)": 22.5,   
    "4. Summer (> 25°C)": 30.0      
}

# Use the time periods as the options list
TIME_OF_DAY_OPTIONS = ['Morning', 'Noon', 'Evening', 'Midnight']

# --- DATA LOADING AND PREDICTION ---

@st.cache_data
def load_data_and_predict():
    """
    Loads historical data, engineers features, loads the LightGBM model, 
    and generates the Predicted_Demand column.
    """
    
    # 1. Load Data
    data_file_path = 'cleaned_energy_weather_data(1).csv' 
    model_file_path = MODEL_FILENAME
    
    try:
        df = pd.read_csv(data_file_path, sep=',') 
    except FileNotFoundError:
        st.error(f"Data file '{data_file_path}' not found.")
        return pd.DataFrame()

    # 2. Basic Feature Engineering (Must match training features)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)
    df['hour'] = df[DATE_COL].dt.hour
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10

    # 3. Time-of-Day and Scenario Mapping
    def map_custom_time_of_day(hour):
        if 0 <= hour <= 5: return 'Midnight'
        elif 6 <= hour <= 11: return 'Morning'
        elif 12 <= hour <= 16: return 'Noon'
        elif 17 <= hour <= 23: return 'Evening'
        else: return 'Other'
            
    df['Detailed_Time_of_Day'] = df['hour'].apply(map_custom_time_of_day)

    def map_temp_to_scenario(temp):
        if temp <= 10: return '1. Cold (≤ 10°C)'
        elif 10 < temp <= 20: return '2. Mild (10°C - 20°C)'
        elif 20 < temp <= 25: return '3. Warm (20°C - 25°C)'
        else: return '4. Summer (> 25°C)'
            
    df['Scenario'] = df[TEMP_CELSIUS_COL].apply(map_temp_to_scenario)
    df = df[df['Detailed_Time_of_Day'].isin(TIME_OF_DAY_OPTIONS)].copy()

    # 4. Prepare Features for Model
    
    # List of numerical/raw features assumed necessary
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
    
    # Handle Categorical Columns with One-Hot Encoding (OHE) as LightGBM expects
    df_features = pd.concat([df_features, pd.get_dummies(df['CountryCode'], prefix='CountryCode', drop_first=True)], axis=1)
    df_features = pd.concat([df_features, pd.get_dummies(df['Detailed_Time_of_Day'], prefix='Detailed_Time_of_Day')], axis=1)
    
    # 5. Load Model and Predict
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_file_path}' not found. Cannot generate predictions.")
        return df

    # Get the feature names the model expects (Safest way)
    try:
        model_feature_names = model.feature_name_
        X_predict = df_features.filter(model_feature_names, axis=1)
        X_predict = X_predict[model_feature_names] # Ensure column order matches
    except AttributeError:
        st.error("Model feature names not found. Skipping prediction.")
        return df

    # Predict
    df[VIS_TARGET_COL] = model.predict(X_predict)
    
    return df

# --- PLOTTING FUNCTIONS ---

@st.cache_data
def create_demand_heatmap(df):
    """Generates a heatmap of Median Predicted Demand vs. Time of Day and Scenario."""
    
    # 1. Aggregate the data: Median Predicted Demand for each combination
    df_agg = df.groupby(['Detailed_Time_of_Day', 'Scenario'])[VIS_TARGET_COL].median().reset_index()
    df_agg = df_agg.rename(columns={VIS_TARGET_COL: 'Median_Predicted_Demand_MW'})
    
    # Define orders
    time_order = ['Morning', 'Noon', 'Evening', 'Midnight']
    scenario_order = list(SCENARIO_MAP.keys())

    # Create the Heatmap (Mark_rect)
    base = alt.Chart(df_agg).encode(
        x=alt.X('Scenario:N', title='Temperature Scenario', sort=scenario_order),
        y=alt.Y('Detailed_Time_of_Day:N', title='Time of Day Period', sort=time_order),
    ).properties(
        title='Median Predicted Demand (MW) by Time of Day and Scenario (Heatmap)'
    )

    # Rectangles (Heatmap colors)
    heatmap = base.mark_rect().encode(
        color=alt.Color('Median_Predicted_Demand_MW:Q', title='Median Predicted Demand (MW)', scale=alt.Scale(range='heatmap'), legend=alt.Legend(orient="top")),
        tooltip=[
            'Detailed_Time_of_Day:N', 
            'Scenario:N', 
            alt.Tooltip('Median_Predicted_Demand_MW:Q', title='Median Predicted Demand', format=',.0f')
        ]
    )

    # Text labels on the heatmap
    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Median_Predicted_Demand_MW:Q', format=',.0f'),
        color=alt.condition(
            alt.datum.Median_Predicted_Demand_MW > df_agg['Median_Predicted_Demand_MW'].quantile(0.7), 
            alt.value('white'),
            alt.value('black')
        )
    )

    chart = heatmap + text
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        """
        _**Summary Heatmap:** This matrix shows the median **Predicted Demand**. It confirms the highest median risk for your early prediction of energy shortages occurs in the **Warm/Summer Evening** and **Cold/Mild Morning** cells._
        """
    )


@st.cache_data
def create_time_of_day_boxplots(df, selected_scenario):
    """Generates four side-by-side box plots for all Time of Day periods, filtered by Scenario, using Predicted Demand."""
    
    df_filtered = df[df['Scenario'] == selected_scenario].copy()
    
    if df_filtered.empty:
        st.warning(f"No data found for scenario '{selected_scenario}'.")
        return

    time_order = ['Morning', 'Noon', 'Evening', 'Midnight']

    chart = alt.Chart(df_filtered).mark_boxplot(extent=1.5).encode(
        # X: Time of Day (the 4 side-by-side plots)
        x=alt.X('Detailed_Time_of_Day:N', title='Time of Day Period', sort=time_order),
        
        # Y: Predicted Demand
        y=alt.Y(VIS_TARGET_COL, title='Predicted Demand (MW) Distribution', scale=alt.Scale(zero=False)),
        
        # Color: Time of Day (for visual distinction of the four plots)
        color=alt.Color('Detailed_Time_of_Day:N', title='Time of Day', sort=time_order),
        
        tooltip=[
            'Detailed_Time_of_Day',
            alt.Tooltip(VIS_TARGET_COL, title='Predicted Demand (MW)', format=',.0f')
        ]
    ).properties(
        title=f'Predicted Demand Distribution by Time of Day (Filtered to: {selected_scenario} Scenario)'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown(f"**Visualization Details:** These four side-by-side box plots show the detailed distribution (quartiles and outliers) of **Predicted Demand** across your custom time periods, specifically when the temperature is in the **{selected_scenario}** range.")

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Energy Demand Prediction Scenario Visualizer")
    st.title("📊 Energy Demand Prediction Scenario Visualizer")
    st.markdown("This dashboard leverages your **LightGBM model** to visualize the predicted energy demand distributions, crucial for your **early prediction of energy shortages in Dutch neighborhoods**.")

    df_predicted = load_data_and_predict()
    
    if df_predicted.empty:
        return

    # ----------------------------------------------------------------------
    # --- COMPARISON PLOT (Heatmap - Captures the matrix logic) ---
    # ----------------------------------------------------------------------
    st.subheader("1. Summary of Median Predicted Demand by Time/Scenario (Heatmap)")
    create_demand_heatmap(df_predicted)
    
    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: SCENARIO SELECTION ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Interactive Filter")
    
    # 1. Scenario Selection 
    selected_scenario = st.sidebar.selectbox(
        "Select Temperature Scenario for Detailed Analysis:",
        options=list(SCENARIO_MAP.keys()),
        index=list(SCENARIO_MAP.keys()).index("3. Warm (20°C - 25°C)"), # Default to Warm
        help="Select a specific temperature scenario to see the detailed Predicted Demand distribution across all four time periods (4 side-by-side box plots)."
    )
    
    st.sidebar.info(f"Viewing detailed prediction distribution for: **{selected_scenario}**")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- DETAILED BOX PLOT (4 Plots Side-by-Side - Now uses Predictions) ---
    # ----------------------------------------------------------------------
    
    st.subheader(f"2. Predicted Demand Distribution by Time of Day (Filtered to: {selected_scenario} Scenario)")
    
    # Call the new detailed plot function
    create_time_of_day_boxplots(df_predicted, selected_scenario)


# Execute the main function
if __name__ == "__main__":
    main()