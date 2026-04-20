import streamlit as st
import pandas as pd

st.title("StressLens")
st.write("A stress detection and monitoring system for cardiac rehabilitation patients.")

st.header("Upload Your Data")
st.write("Here is where you can upload your data associated from your Empatica E4, or any other device that measures these signals.")
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

with st.sidebar:
    with st.container():
        st.header("What is StressLens?")
        st.write("StressLens acts as an automatic stress logger and reporter to aid Stress Management Therapy for cardiac rehabilitation paitents. It " \
        "automatically detects when you are stressed based on Empatica E4-collected physiological signals and allows the user to fill out a questionairre, " \
        "inspired by current stress-logs (such as University of Ottowa Heart Institute's Stress Management guide).")

        st.header("How does it help?")
        st.write("For Cardiac Rehabilitation Patients that go through Stress Management Training or Therapy, this application acts as an aid to the process " \
        "by automatically tracking and providing data to the patient and caremanager, enabling both parties with crucial information to make better"
        "decisions for the rehabilitation process.")
        st.markdown("- Automatic stress logger to track most common stressors, physical and mental symptoms without the need for a physical tracker.")
        st.markdown("- Tracks the effectiveness of relaxation techniques utilized by the user")
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
        st.write("After uploading the relevant Empatica E4 data, this data is run through a machine learning model to identify when you are stressed." \
        "Stress periods are recorded and create questionaires for each identified time for the user to answer." \
        "After all stress period questionaires are completed, the report is generated for that time period.")

