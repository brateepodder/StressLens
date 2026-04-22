import streamlit as st

def render_episode_forms(episodes):
    if not episodes:
        st.info("No stress episodes detected for the data uploaded.")
        return

    st.header("🔴 Stress Detected")
    st.write(f"We detected **{len(episodes)}** episodes. Please provide context for each.")

    for i, ep in enumerate(episodes):
        # Create a unique key for each form
        with st.form(key=f"stress_form_{i}"):
            st.subheader(f"Episode {i+1}: {ep['start_iso']} to {ep['end_iso']}")
            st.caption(f"Duration: {ep['duration_sec']} seconds")

            # --- TRIGGER SECTION ---
            trigger = st.multiselect(
                "What triggered this stress?",
                ["Daily Hassle", "Commute", "Work", "Family", "Physical", "Life Event", "Financial", "Thoughts", "Other"],
                key=f"trigger_{i}"
            )
            
            # --- EMOTIONAL SYMPTOMS ---
            emotions = st.multiselect(
                "How were you feeling?",
                ["Motivated", "Calm", "Happy", "Sad", "Scared/Anxious", "Numb", "Angry"],
                key=f"emotion_{i}"
            )
            
            intensity = st.slider("Intensity of symptoms (1-5)", 1, 5, 3, key=f"intensity_{i}")

            # --- BEHAVIORAL INTERVENTION ---
            action = st.selectbox(
                "What was your behavioral response?",
                [
                    "Deep breathing", "Visualization", "Meditation", 
                    "Progressive Muscle Relaxation", "Stretching", 
                    "Self-massage", "Talk to a individual", 
                    "Distraction (Media/Music/Hobby)", "Exercise", "Other"
                ],
                key=f"action_{i}"
            )
            
            success_rate = st.select_slider(
                "How successful was this intervention?",
                options=[1, 2, 3, 4, 5],
                key=f"success_{i}"
            )

            feedback = st.text_area("How did you feel after? (Plain language)", key=f"feedback_{i}")

            # --- SUBMIT ---
            submitted = st.form_submit_button("Save Reflection")
            if submitted:
                # Logic to save this specific episode reflection to your database/CSV
                # save_reflection_to_db(ep, trigger, emotions, intensity, action, success_rate, feedback)
                st.success(f"Reflection for Episode {i+1} saved!")