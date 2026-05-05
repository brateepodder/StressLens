import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.preprocessing import preprocessing_pipeline
from pathlib import Path
from datetime import datetime

if "completed_episodes" not in st.session_state:
    st.session_state.completed_episodes = set()

if "reflections" not in st.session_state:
    st.session_state.reflections = {} 

# PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressLens",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("StressLens")
st.write("A stress detection and monitoring system for cardiac rehabilitation patients.")

# UPLOADING DATA ──────────────────────────────────────────────────────────────
st.header("Upload Your Data")
st.write("Here is where you can upload your data associated from your Empatica E4, or any other " \
"device that measures these signals.")

# SIDEBAR (FOR INFORMATION) ──────────────────────────────────────────────────────────────
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

# DATA SUBMISSION FORM ──────────────────────────────────────────────────────────────
with st.form("data_upload_form"):
    st.subheader("Empatica E4 Data Upload")
    
    acc_file = st.file_uploader("Upload Accelerometer (ACC.csv)")
    bvp_file = st.file_uploader("Upload BVP (BVP.csv)")
    eda_file = st.file_uploader("Upload EDA (EDA.csv)")
    temp_file = st.file_uploader("Upload Temperature (TEMP.csv)")
    
    submit_button = st.form_submit_button("Start Processing")

# DATA SUBMISSION BUTTON ──────────────────────────────────────────────────────────────
if submit_button:
    if all([acc_file, bvp_file, eda_file, temp_file]):
        with st.spinner("Processing biometric data..."):
            episodes, results_df = preprocessing_pipeline(acc_file, bvp_file, eda_file, temp_file)
            st.success("Processing Complete!")

            # EPISODES FORM MANAGEMENT 
            st.session_state.episodes = episodes
            st.session_state.results_df = results_df
            st.session_state.completed_episodes = set()


    else:
        st.error("Please upload all four files before submitting.")

# QUESTIONAIRRE RENDERER ──────────────────────────────────────────────────────────────
def render_episode_forms(episodes):
    remaining_episodes = [
        (i, ep) for i, ep in enumerate(episodes) 
        if i not in st.session_state.completed_episodes
    ]

    if not remaining_episodes:
        if "episodes" in st.session_state:
            st.success("🎉 All stress episodes have been reviewed! Generating report...")
            generate_care_manager_report()
        return

    st.header("🔴 Stress Detected")
    st.write(f"You have **{len(remaining_episodes)}** episodes left to review.")

    # FORM CONTAINER 
    with st.container(height=500, border=True):
        for i, ep in remaining_episodes:
            top_physiological_response = ep['leading_factor']['display_name']
            start_dt = datetime.fromtimestamp(ep['start_unix'])
            end_dt = datetime.fromtimestamp(ep['end_unix'])
            readable_start = start_dt.strftime("%d %B %Y, %H:%M")
            readable_end = end_dt.strftime("%H:%M") 

            with st.form(key=f"stress_form_{i}"):
                st.subheader(f"Episode {i+1}: {readable_start} to {readable_end}")
                st.caption(
                f"Duration: {ep['duration_sec'] // 60}m {ep['duration_sec'] % 60}s  •  "
                f"Top physiological symptom: {top_physiological_response}"
            )

                # --- CORRECT/INCORRECT CLASSIFICATION ---
                classification = st.selectbox(
                    "Are you stressed?",
                    ("Yes", "No", "Ignore"),
                )

                # --- TRIGGER SECTION ---
                trigger = st.multiselect(
                    "What triggered the stress?",
                    ["Daily Hassle", "Commute", "Work", "Family", "Physical", "Life Event", 
                    "Financial", "Thoughts", "Other"],
                    key=f"trigger_{i}"
                )
                
                # --- EMOTIONAL SYMPTOMS ---
                emotions = st.multiselect(
                    "How are you feeling?",
                    ["Motivated", "Calm", "Happy", "Sad", "Scared/Anxious", "Numb", "Angry"],
                    key=f"emotion_{i}"
                )
                
                # --- BEHAVIORAL INTERVENTION ---
                action = st.selectbox(
                    "What is your response?",
                    [
                        "Deep breathing", "Visualization", "Meditation", 
                        "Progressive Muscle Relaxation", "Stretching", 
                        "Self-massage", "Talk to a individual", 
                        "Distraction (Media/Music/Hobby)", "Exercise", "Other"
                    ],
                    key=f"action_{i}"
                )
                
                success_rate = st.select_slider(
                    "How successful was this intervention to you?",
                    options=[1, 2, 3, 4, 5],
                    key=f"success_{i}"
                )

                feedback = st.multiselect(
                    "How did you feel after?",
                    ["Less Stressed", "Same Amount of Stressed", "More Stressed"],
                    key=f"feedback_{i}"
                )

                    
                submitted = st.form_submit_button("Save Reflection")
                
                if submitted:
                        st.session_state.reflections[i] = {
                            "original_episode": ep,
                            "classification": classification,
                            "triggers": trigger,
                            "emotions": emotions,
                            "action": action,
                            "success_rate": success_rate,
                            "feedback": feedback,
                            "duration": ep['duration_sec']
                        }
                        st.session_state.completed_episodes.add(i)
                        
                        # DATABASE SAVE
                        
                        # FORCE RERUN FOR FORM DISAPPEARANCE
                        st.rerun()

# RENDERING QUESTIONAIRRES ──────────────────────────────────────────────────────────────
if "episodes" in st.session_state:
    render_episode_forms(st.session_state.episodes)

# GENERATING REPORT ──────────────────────────────────────────────────────────────
def generate_care_manager_report():
    st.divider()
    st.header("Care Manager Report")
    
    ref_list = list(st.session_state.reflections.values())
    if not ref_list:
        st.warning("No data available to generate report.")
        return
        
    df_ref = pd.DataFrame(ref_list)
    df_confirmed = df_ref[df_ref['classification'] == "Yes"].copy()

    # ── SUMMARY METRICS ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stress Episodes", len(df_confirmed))
    with col2:
        avg_dur = df_confirmed['duration'].mean() if not df_confirmed.empty else 0
        st.metric("Avg. Duration", f"{int(avg_dur // 60)}m {int(avg_dur % 60)}s")
    with col3:
        std_dur = df_confirmed['duration'].std() if len(df_confirmed) > 1 else 0
        st.metric("Std Dev Duration", f"{std_dur:.1f}s")

    # ── TRENDS & INSIGHTS ─────────────────────────────────────────────────────
    st.subheader("Trends & Insights")
    c1, c2 = st.columns(2)

    with c1:
        st.write("**Top Stressors**")
        if not df_confirmed.empty:
            triggers_exploded = df_confirmed['triggers'].explode()
            triggers_exploded = triggers_exploded[triggers_exploded.notna() & (triggers_exploded != "")]

            if not triggers_exploded.empty:
                trigger_stats = (
                    triggers_exploded
                    .to_frame(name="trigger")
                    .join(df_confirmed[["duration"]])
                    .groupby("trigger")["duration"]
                    .agg(Count="count", Avg_Duration_sec="mean")
                    .sort_values("Count", ascending=False)
                    .head(5)
                )
                trigger_stats["Avg Duration"] = trigger_stats["Avg_Duration_sec"].apply(
                    lambda s: f"{int(s // 60)}m {int(s % 60)}s"
                )
                st.table(trigger_stats[["Count", "Avg Duration"]])
            else:
                st.write("No triggers recorded.")
        else:
            st.write("No confirmed episodes.")

    with c2:
        st.write("**Top Techniques Used**")
        if not df_confirmed.empty:
            technique_stats = (
                df_confirmed
                .groupby("action")["duration"]
                .agg(Count="count", Avg_Duration_sec="mean")
                .sort_values("Count", ascending=False)
                .head(5)
            )
            technique_stats["Avg Duration"] = technique_stats["Avg_Duration_sec"].apply(
                lambda s: f"{int(s // 60)}m {int(s % 60)}s"
            )
            st.table(technique_stats[["Count", "Avg Duration"]])
        else:
            st.write("No actions recorded.")

    # --- BASELINE NOTE ---
    st.info("Still need to add the following: " \
    "1. Keeping track of best relaxation techniques for recommendation by storing all response inputs, " \
    "2. Intensity markers for each emotional symptom before and after episode" \
    "3. Finish all information to be included in Care Report, including showing raw data.")