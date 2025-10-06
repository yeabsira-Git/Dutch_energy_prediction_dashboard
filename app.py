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
        # FIX: Explicitly setting sep=',' to prevent EmptyDataError due to delimiter inference issues.
        df = pd.read_csv(file_path, sep=',') 
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

# --- PLOTTING FUNCTIONS ---

@st.cache_data
def create_comparison_plot(df):
    """
    Generates a bar chart showing Median Demand for every Time of Day / Scenario 
    combination, with critical scenarios highlighted.
    """
    
    # 1. Aggregate the data: Median Demand for each combination
    df_agg = df.groupby(['Detailed_Time_of_Day', 'Scenario'])[TARGET_COL].median().reset_index()
    df_agg = df_agg.rename(columns={TARGET_COL: 'Median_Demand_MW'})
    
    # --- HIGHLIGHTING LOGIC ---
    def get_highlight_group(row):
        scenario = row['Scenario']
        time = row['Detailed_Time_of_Day']
        
        # Highlight Warm/Summer Evening spikes (User's request)
        if time == 'Evening' and ('Warm' in scenario or 'Summer' in scenario):
            return '🔥 Critical Summer/Warm Evening Spike'
        # Highlight Cold/Mild Morning spikes (Initial observation)
        elif time == 'Morning' and ('Cold' in scenario or 'Mild' in scenario):
            return '🚨 Critical Winter/Mild Morning Spike'
        else:
            return 'Baseline Scenario'

    df_agg['Highlight_Group'] = df_agg.apply(get_highlight_group, axis=1)
    
    # Define colors for the highlight groups
    highlight_color_scale = alt.Scale(
        domain=['🔥 Critical Summer/Warm Evening Spike', '🚨 Critical Winter/Mild Morning Spike', 'Baseline Scenario'],
        range=['#FF4B4B', '#FF8C00', '#ADD8E6'] # Red, Orange, Light Blue
    )
    # ---------------------------

    # Sort the Time of Day and Scenario for consistent plotting order
    time_order = ['Midnight', 'Morning', 'Noon', 'Evening']
    scenario_order = list(SCENARIO_MAP.keys())
    
    # 2. Create the Faceted Bar Chart
    chart = alt.Chart(df_agg).mark_bar().encode(
        # X: Scenario
        x=alt.X('Scenario:N', title='Temperature Scenario', sort=scenario_order, axis=alt.Axis(labels=False)),
        
        # Y: Median Demand
        y=alt.Y('Median_Demand_MW:Q', title='Median Historical Demand (MW)', scale=alt.Scale(zero=False)),
        
        # Color: Use the new Highlight Group
        color=alt.Color('Highlight_Group:N', title='Scenario Peak Focus', scale=highlight_color_scale),
        
        # Column: Time of Day (Faceting)
        column=alt.Column('Detailed_Time_of_Day:N', 
                         header=alt.Header(titleOrient="bottom", labelOrient="bottom"), 
                         sort=time_order, 
                         title="Time of Day Period"),
                         
        # Tooltip for details
        tooltip=[
            'Detailed_Time_of_Day:N', 
            'Scenario:N', 
            alt.Tooltip('Median_Demand_MW:Q', title='Median Demand', format=',.0f'),
            'Highlight_Group:N'
        ]
    ).properties(
        title='Median Demand Spikes Across Time of Day and Scenario (Critical Periods Highlighted)'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        """
        _This visual enhancement clearly highlights the **Warm/Summer Evening** demand spike 
        (**🔥 Red**) and the **Cold/Mild Morning** spike (**🚨 Orange**), which are critical inputs for your energy shortage prediction model._
        """
    )


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
    st.markdown(f"**Visualization Details:** The box plot shows the granular distribution (quartiles, outliers) of energy demand shifts across temperature scenarios specifically during the **{selected_time_of_day}** period.")

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Historical Demand Scenario Visualizer")
    st.title("📊 Historical Energy Demand Scenario Visualizer")
    st.markdown("This dashboard provides the granular historical context necessary for your **early prediction of energy shortages in Dutch neighborhoods**.")

    df_historical = load_historical_data()
    
    if df_historical.empty:
        # Error handling will be displayed by the load function
        return

    # ----------------------------------------------------------------------
    # --- COMPARISON PLOT (Summary of Spikes) ---
    # ----------------------------------------------------------------------
    st.subheader("1. Summary of Demand Spikes (Median Demand)")
    create_comparison_plot(df_historical)
    
    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- INTERACTIVE INPUTS: TIME OF DAY SELECTION ---
    # ----------------------------------------------------------------------
    st.sidebar.header("Interactive Filter")
    
    # 1. Time of Day Selection
    selected_time_of_day = st.sidebar.selectbox(
        "Select Time of Day Period:",
        options=TIME_OF_DAY_OPTIONS,
        index=TIME_OF_DAY_OPTIONS.index('Noon'), # Default to Noon
        help="Select a specific time period to see the detailed demand distribution across the four temperature scenarios."
    )
    
    st.sidebar.info(f"Viewing detailed distribution for: **{selected_time_of_day}**")

    st.markdown("---")
    
    # ----------------------------------------------------------------------
    # --- DETAILED BOX PLOT ---
    # ----------------------------------------------------------------------
    
    st.subheader(f"2. Detailed Demand Distribution by Scenario (Filtered to: {selected_time_of_day} Period)")
    
    # Call the filtered plot function
    create_filtered_boxplot(df_historical, selected_time_of_day)


# Execute the main function
if __name__ == "__main__":
    main()