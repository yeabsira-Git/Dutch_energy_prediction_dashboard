import os
import pandas as pd
import streamlit as st

# The relative path that should work based on your GitHub structure
RELATIVE_PATH = 'data/historical_data.csv' 

# Get the absolute path it is looking for
full_path = os.path.abspath(RELATIVE_PATH)

st.info(f"The app is looking for the file at: {full_path}")

try:
    data = pd.read_csv(RELATIVE_PATH)
    st.success("Data loaded successfully!")

except FileNotFoundError:
    st.error(f"Failed to find file at: {full_path}")
    st.error(f"Current Working Directory: {os.getcwd()}")
    st.error("Please check the full path output and ensure your CSV is there on GitHub.")
    # Re-raise the error so the app fails gracefully
    st.stop()
