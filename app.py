import streamlit as st
import pandas as pd

st.title("StressLens")
st.write("A stress detection and monitoring system for cardiac rehabilitation patients")

with st.container():
    st.header("What is StressLens")
    st.write("StressLens acts as an automatic stress logger and reporter to aid Stress Management Therapy for cardiac rehabilitation paitents. It " \
    "automatically detects when you are stressed based on Empatica E4-collected physiological signals and allows the user to fill out a questionairre, " \
    "inspired by current stress-logs, such as University of Ottowa Heart Institute's Stress Management guide. This allows for caregivers to track patient's" \
    "progress on using certain relaxation techniques, which triggers are most prominent and require intervention, and signal if the patient has low HRV " \
    "indices tied to higher mortality rates.")

with st.sidebar:
    ACC = st.file_uploader("Upload the accelerometer data from Empatica E4.")
    if ACC is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(ACC)
        st.write(dataframe)

    BVP = st.file_uploader("Upload the BVP data from Empatica E4.")
    if BVP is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(BVP)
        st.write(dataframe)

    EDA = st.file_uploader("Upload the EDA data from Empatica E4.")
    if EDA is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(EDA)
        st.write(dataframe)
    
    TEMP = st.file_uploader("Upload the temperature data from Empatica E4.")
    if TEMP is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(TEMP)
        st.write(dataframe)

