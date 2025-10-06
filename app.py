import streamlit as st
import pandas as pd
import numpy as np
import altair as alt 
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TARGET_COL = 'Demand_MW'
TEMP_COL = 'Temperature (0.1 degrees Celsius)'
TEMP_CELSIUS_COL = 'Temperature_C' 
DATE_COL = 'DateUTC'

# --- SCENARIO MAPPING ---
SCENARIO_MAP = {
    "1. Cold (≤ 10°C)": 5.0,     
    "2. Mild (10°C - 20°C)": 15.0,   
    "3. Warm (20°C - 25°C)": 22.5,   
    "4. Summer (> 25°C)": 30.0      
}

# Use the time periods as the options list
TIME_OF_DAY_OPTIONS = ['Morning', 'Noon', 'Evening', 'Midnight']

# --- DATA LOADING ---

@st.cache_data
def load_historical_data():
    """
    Loads historical data, calculates temperature in Celsius, maps scenarios, 
    and applies the custom Time-of-Day mapping logic.
    """
    file_path = 'cleaned_energy_weather_data(1).csv' 
    
    try:
        # FIX: Added encoding='latin1' to prevent EmptyDataError due to parsing issues
        df = pd.read_csv(file_path, encoding='latin1') 
    except FileNotFoundError:
        st.error("Historical data file 'cleaned_energy_weather_data(1).csv' not found.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file was found but appears to be empty or unreadable (EmptyDataError).")
        return pd.DataFrame()

    # Convert DateUTC to datetime and extract the hour (assuming UTC time is used consistently)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True)
    df['hour'] = df[DATE_COL].dt.hour
    
    # Calculate temperature in Celsius
    df[TEMP_CELSIUS_COL] = df[TEMP_COL] / 10
    
    # 1. CUSTOM TIME-OF-DAY MAPPING FUNCTION
    def map_custom_time_of_day(hour):
        """Maps the hour (0-23) to the custom time periods."""
        if 0 <= hour <= 5:  # 0:00 to 5:59
            return 'Midnight'
        elif 6 <= hour <= 11:  # 6:00 to 11:59
            return 'Morning'
        elif 12 <= hour <= 16:  # 12:00 to 16:59
            return 'Noon'
        elif 17 <= hour <= 23:  # 17:00 to 23:59
            return 'Evening'
        else:
            return 'Other'
            
    # Apply the custom mapping, overwriting any previous 'Detailed_Time_of_Day' value
    df['Detailed_Time_of_Day'] = df['hour'].apply(map_custom_time_of_day)

    # 2. SCENARIO MAPPING FUNCTION
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
            
    df['Scenario'] = df[TEMP_CELSIUS_COL].apply(map_temp_to_scenario)

    # Filter out any 'Other' periods just in case (e.g., if there were missing hour data)
    df = df[df['Detailed_Time_of_Day'].isin(TIME_OF_DAY_OPTIONS)]
    
    return df

# --- PLOTTING FUNCTION ---

@st.cache_data
def create_filtered_boxplot(df, selected_time_of_day):
    """Generates a box plot for Demand vs. Scenario, filtered by Time of Day."""
    
    df_filtered = df[df['Detailed_Time_of_Day'] == selected_time_of_day]

    if df_filtered.empty:
        st.warning(f"No historical data found for '{selected_time_of_day}'.")
        return

    # Create the single Box Plot (Demand vs. Scenario)
    chart = alt.Chart(df_filtered).mark_boxplot(extent=1.5).encode(
        x=alt.X('Scenario:N', title='Temperature Scenario', sort=list(SCENARIO_MAP.keys())),
        y=alt.Y(TARGET_COL, title='Historical Demand (MW)', scale=alt.Scale(zero=False)),
        color=alt.Color('Scenario:N', title='Scenario', sort=list(SCENARIO_MAP.keys())),
        tooltip=[
            'Scenario',
            alt.Tooltip(TARGET_COL, title='Demand (MW) Distribution', format=',.0f')
        ]
    ).properties(
        title=f'Demand Distribution by Scenario (Filtered to: {selected_time_of_day})'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown(f"**Visualization Details:** The box plot shows how the distribution of energy demand shifts across temperature scenarios specifically during the **{selected_time_of_day}** period.")

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Historical Demand Scenario Visualizer")
    st.title("📊 Historical Energy Demand Scenario Visualizer")
    st.markdown("Use the controls on the left to analyze how demand changes by temperature scenario during key times of the day based on your custom time periods.")

    df_historical = load_historical_data()
    
    if df_historical.empty:
        # Error handling will be displayed by the load function
        return

    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: TIME OF DAY SELECTION ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Interactive Filter")
    
    # 1. Time of Day Selection (Replaces Scenario Selection)
    selected_time_of_day = st.sidebar.selectbox(
        "Select Time of Day Period:",
        options=TIME_OF_DAY_OPTIONS,
        index=TIME_OF_DAY_OPTIONS.index('Noon'), # Default to Noon
        help="Select a specific time period to see how demand is distributed across the different temperature scenarios during that time."
    )
    
    st.sidebar.info(f"Using Custom Period: **{selected_time_of_day}**")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- DASHBOARD PLOT (The only visualization) ---
    # ----------------------------------------------------------------------
    
    st.subheader(f"Demand Distribution by Temperature Scenario during the {selected_time_of_day} Period")
    
    # Call the filtered plot function
    create_filtered_boxplot(df_historical, selected_time_of_day)

    st.markdown("---")
    st.markdown(f"This focused dashboard provides the granular historical context needed for your project on **early prediction of energy shortages** in Dutch neighborhoods, clearly showing the interplay between temperature and your **custom time-of-day periods**.")


# Execute the main function
if __name__ == "__main__":
    main()