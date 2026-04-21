import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.preprocessing import preprocessing

from pathlib import Path

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressLens",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── TITLE ──────────────────────────────────────────────────────────────
st.title("StressLens")
st.write("A stress detection and monitoring system for cardiac rehabilitation patients.")

# ── UPLOADING DATA ──────────────────────────────────────────────────────────────
st.header("Upload Your Data")
st.write("Here is where you can upload your data associated from your Empatica E4, or any other " \
"device that measures these signals.")

# ── MODEL LOADER ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading stress detection model…")
def load_model():
    """
    Loads the trained model from model/stress_model.joblib.
    Cache keeps it in memory so it is only loaded once per session.
    """
    model_path = Path(__file__).parent / "model" / "stress_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Make sure stress_model.joblib is inside the 'model/' folder."
        )
    return joblib.load(model_path)

# SIDEBAR (FOR INFORMATION)
with st.sidebar:
    with st.container():
        st.header("What is StressLens?")
        st.write("StressLens acts as an automatic stress logger and reporter to aid Stress Management Therapy for cardiac rehabilitation patients. It " \
        "automatically detects when you are stressed based on Empatica E4-collected physiological signals and allows the user to fill out a questionairre, " \
        "inspired by current stress-logs (such as University of Ottowa Heart Institute's Stress Management guide).")

        st.header("How does it help?")
        st.write("For Cardiac Rehabilitation Patients that go through Stress Management Training or Therapy, this application acts as an aid to the process " \
        "by automatically tracking and providing data to the patient and caremanager, enabling both parties with crucial information to make better"
        "decisions for the rehabilitation process.")
        st.markdown("- Automatic stress logger to track most common stressors, physical and mental symptoms without the need for a physical tracker.")
        st.markdown("- Tracks the effectiveness of relaxation techniques utilized by the user by how quickly it calms the user down")
        st.markdown("- Tracks important HRV metrics associated with high mortality for patients that had cardiac events")
        st.markdown("- Allows the patient and caregiver to view the effect of their progress and therapy with real data")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)

        st.header("How does StressLens Work?")
        st.write("After uploading the relevant Empatica E4 data, this data is run through a machine learning model to identify when you are stressed. " \
        "Stress periods are recorded and create questionaires for each identified time for the user to answer. " \
        "After all stress period questionaires are completed, the report is generated for that time period.")

# FORM & FUNCTION FOR SUBMITTING 
with st.form("data_upload_form"):
    st.subheader("Empatica E4 Data Upload")
    
    acc_file = st.file_uploader("Upload Accelerometer (ACC.csv)")
    bvp_file = st.file_uploader("Upload BVP (BVP.csv)")
    eda_file = st.file_uploader("Upload EDA (EDA.csv)")
    temp_file = st.file_uploader("Upload Temperature (TEMP.csv)")
    
    submit_button = st.form_submit_button("Start Processing")

if submit_button:
    if all([acc_file, bvp_file, eda_file, temp_file]):
        with st.spinner("Processing biometric data..."):
            # Pass the file objects directly to your src function
            processed_df = preprocessing_pipeline(acc_file, bvp_file, eda_file, temp_file)
            st.success("Processing Complete!")
            st.write(processed_df)
    else:
        st.error("Please upload all four files before submitting.")