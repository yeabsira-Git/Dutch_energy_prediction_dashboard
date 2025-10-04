# app.py: Streamlit Dashboard Frontend

import streamlit as st
import plotly.express as px
import pandas as pd
# FIX: Using the correct import now that the file exists
from energy_predictor import load_and_preprocess_data, generate_forecast

# --- 2. CONFIGURATION AND DATA LOAD ---
@st.cache_data
def get_data():
    return load_and_preprocess_data()

st.set_page_config(layout="wide", page_title="Dutch Energy Shortage Prediction")

data = get_data()

if data.empty:
    st.error("🛑 CRITICAL ERROR: Could not load historical data. Ensure the CSV file is in your GitHub.")
    st.stop()


# --- 3. PAGE TITLE AND SIDEBAR (USER INPUT) ---
st.title("🇳🇱 Early Prediction of Energy Shortages")
st.markdown("Use the controls to adjust the forecast and capacity threshold.")

st.sidebar.header("Dashboard Controls")

n_hours_forecast = st.sidebar.slider(
    'Forecast Horizon (Hours)',
    min_value=24, max_value=168, value=72, step=24
)

demand_threshold = st.sidebar.slider(
    'Set Energy Supply Capacity Threshold (MW)',
    min_value=int(data['Demand_MW'].min() * 0.9),
    max_value=int(data['Demand_MW'].max() * 1.1),
    value=1800, 
    step=50
)


# --- 4. FORECAST GENERATION AND PLOT ---
st.header("1. Model Forecast and Capacity Alert")

forecast_df = generate_forecast(data, n_hours=n_hours_forecast)

if not forecast_df.empty:
    fig_forecast = px.line(
        forecast_df,
        y='Predicted_Demand_MW',
        title=f'Predicted Energy Demand (MW) for the next {n_hours_forecast} hours',
        labels={'Predicted_Demand_MW': 'Demand (MW)'},
        template='plotly_white',
    )
    
    fig_forecast.add_hline(
        y=demand_threshold, 
        line_dash="dash", 
        annotation_text="Supply Capacity Limit", 
        annotation_position="top right", 
        line_color="red"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)
    
    shortage_hours = forecast_df[forecast_df['Predicted_Demand_MW'] > demand_threshold]
    if not shortage_hours.empty:
        st.error(f"⚠️ **SHORTAGE ALERT:** {len(shortage_hours)} hours are predicted to exceed the {demand_threshold} MW capacity limit.")
    else:
        st.success("✅ Forecast is currently below the supply capacity limit.")


# --- 5. DRIVER ANALYSIS PLOT ---
st.header("2. Key Driver Analysis: Demand vs. Temperature by Hour")

fig_drivers = px.scatter(
    data,
    x='Temperature_C',
    y='Demand_MW',
    color='Hour',
    color_continuous_scale=px.colors.sequential.Inferno,
    facet_col='Temperature_Category',
    facet_col_wrap=2,
    title='Energy Demand vs. Temperature: Detailed Hourly Relationship',
    template='plotly_white',
    height=500
)

st.plotly_chart(fig_drivers, use_container_width=True)
