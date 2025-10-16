import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import warnings
warnings.filterwarnings("ignore")

# --- Load Model and Data
model = joblib.load("best_model.pkl")
df = pd.read_csv("student_habits_performance_updated.csv")
feature_names = ["study_hours_per_day", "attendance_percentage", "mental_health_rating", "sleep_hours"]

st.set_page_config(page_title="Student Performance Dashboard", page_icon="📊", layout="wide")
st.title("Student Performance & Explainability Dashboard")

# SIDEBAR for student selection and adjusting inputs
with st.sidebar:
    st.header("Student Profile & Simulation")
    student_ids = df["student_id"].unique()
    selected_id = st.selectbox("Select Student", student_ids)
    student_row = df[df["student_id"] == selected_id].iloc[0]
    st.subheader("Adjust Input Values")
    study_hours = st.slider("Study Hours per Day", 0, 12, int(student_row["study_hours_per_day"]))
    attendance = st.slider("Attendance Percentage", 0, 100, int(student_row["attendance_percentage"]))
    sleep_hours = st.slider("Sleep Hours per Night", 0, 12, int(student_row["sleep_hours"]))
    mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, int(student_row["mental_health_rating"]))
    run_prediction = st.button("Predict & Explain")

# MAIN BODY
st.markdown("### Student Details")
with st.expander("View Detailed Profile"):
    profile1, profile2 = st.columns(2)
    with profile1:
        st.write(f"**Age:** {student_row['age']}")
        st.write(f"**Gender:** {student_row['gender']}")
        st.write(f"**Diet Quality:** {student_row['diet_quality']}")
        st.write(f"**Exercise Frequency:** {student_row['exercise_frequency']}/week")
        st.write(f"**Extracurricular:** {student_row['extracurricular_participation']}")
    with profile2:
        st.write(f"**Study Hours:** {student_row['study_hours_per_day']} hrs/day")
        st.write(f"**Attendance:** {student_row['attendance_percentage']}%")
        st.write(f"**Sleep:** {student_row['sleep_hours']} hrs/night")
        st.write(f"**Mental Health:** {student_row['mental_health_rating']}/10")
        st.write(f"**Internet Quality:** {student_row['internet_quality']}")
        st.write(f"**Parental Education:** {student_row['parental_education_level']}")
    score1, score2, score3, score4 = st.columns(4)
    score1.metric("Python Score", f"{student_row['python_marks']}")
    score2.metric("Mathematics", f"{student_row['mathematics_marks']}")
    score3.metric("DBMS", f"{student_row['dbms_marks']}")
    score4.metric("Overall Exam", f"{student_row['exam_score']}%")

st.divider()

if run_prediction:
    # Prepare input for prediction
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours]])
    pred = model.predict(input_data)[0]
    pred = np.clip(pred, 0, 100)
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric("Predicted Exam Score", f"{pred:.2f} %")
    with kpi_col2:
        st.metric("Deviation from Actual", f"{pred-student_row['exam_score']:.2f} %")
    st.divider()
    st.subheader("Prediction Explainability")
    background = df[feature_names].sample(100, random_state=42).values
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(input_data)

    # Bar chart
    st.markdown("**Feature Importance (Bar Plot):**")
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
    shap.plots.bar(shap.Explanation(
        values=shap_values.values[0],
        base_values=explainer.expected_value,
        data=input_data[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig_bar)

    # Force plot
    st.markdown("**Feature Impact (Force Plot):**")
    st_shap(shap.plots.force(explainer.expected_value, shap_values.values[0], feature_names=feature_names))

    # Guidance
    st.info(
        "The **bar plot** shows which factors affected this prediction the most. Positive bars raise your score; negative bars reduce it. "
        "The **force plot** explains how each input pushed your prediction up or down from the average, step by step."
    )

