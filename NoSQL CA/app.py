# app.py
# Phase 6: Streamlit web app for the Early Warning System.
# Run it from the project folder:  streamlit run app.py
#
# Pages:
#   1. Dashboard          -> overall picture, read live from MongoDB
#   2. At-Risk Students   -> filter students by subject and risk level
#   3. Student Lookup     -> one student's subjects, actual vs predicted
#   4. Predict New Student-> enter marks after the midterm, get the risk, save to MongoDB
#   5. Model Performance  -> metrics and graphs from Phase 5

import sys
import json
from datetime import datetime
import pandas as pd
import streamlit as st

sys.path.append("src")                     # so we can import our own scripts from src/
from db import get_collection
from predict import predict_subject, SUBJECTS

st.set_page_config(page_title="Student Early Warning System", page_icon="🎓", layout="wide")

students = get_collection()                # the 'students' collection in MongoDB
predictions = students.database["predictions"]   # new predictions made in the app are saved here

RISK_ICONS = {"High Risk": "🔴 High Risk", "Borderline": "🟡 Borderline", "Safe": "🟢 Safe"}

st.sidebar.title("🎓 Early Warning System")
page = st.sidebar.radio("Go to", ["Dashboard", "At-Risk Students", "Student Lookup",
                                  "Predict New Student", "Model Performance"])
st.sidebar.caption("Predicts each student's risk right after the midterm, "
                   "using MongoDB + Linear and Logistic Regression.")


# ---------------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------------
if page == "Dashboard":
    st.title("Dashboard")

    # Count subject results in each risk level ($unwind + $group)
    pipeline = [{"$unwind": "$subjects"},
                {"$group": {"_id": "$subjects.risk_level", "count": {"$sum": 1}}}]
    level_counts = {row["_id"]: row["count"] for row in students.aggregate(pipeline)}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", students.count_documents({}))
    col2.metric("🔴 High Risk results", level_counts.get("High Risk", 0))
    col3.metric("🟡 Borderline results", level_counts.get("Borderline", 0))
    col4.metric("🟢 Safe results", level_counts.get("Safe", 0))

    st.metric("Students at High Risk in 2 or more subjects",
              students.count_documents({"risk_subjects": {"$gte": 2}}))

    # Subject-wise summary
    st.subheader("High Risk results by subject")
    pipeline = [{"$unwind": "$subjects"},
                {"$group": {"_id": "$subjects.subject",
                            "High Risk": {"$sum": "$subjects.at_risk"},
                            "Avg attendance %": {"$avg": "$subjects.attendance_pct"},
                            "Avg midterm (/20)": {"$avg": "$subjects.midterm"}}},
                {"$sort": {"High Risk": -1}}]
    subj = pd.DataFrame(list(students.aggregate(pipeline))).rename(columns={"_id": "Subject"})
    st.bar_chart(subj, x="Subject", y="High Risk", color="#d03b3b")
    st.dataframe(subj.round(1), hide_index=True, width="stretch")

    # Risk by number of reappears
    st.subheader("High Risk % by number of past reappears")
    pipeline = [{"$unwind": "$subjects"},
                {"$group": {"_id": "$reappears", "total": {"$sum": 1},
                            "at_risk": {"$sum": "$subjects.at_risk"}}},
                {"$sort": {"_id": 1}}]
    rep = pd.DataFrame(list(students.aggregate(pipeline)))
    rep["High Risk %"] = (rep["at_risk"] / rep["total"] * 100).round(1)
    rep["Reappears"] = rep["_id"].astype(str)
    st.bar_chart(rep, x="Reappears", y="High Risk %", color="#2a78d6")


# ---------------------------------------------------------------
# Page 2: At-Risk Students
# ---------------------------------------------------------------
elif page == "At-Risk Students":
    st.title("At-Risk Students")

    col1, col2 = st.columns(2)
    subject = col1.selectbox("Subject", ["All"] + SUBJECTS)
    level = col2.selectbox("Risk level", ["High Risk", "Borderline", "Safe"])

    # Build the MongoDB query from the filters
    match = {"subjects.risk_level": level}
    if subject != "All":
        match["subjects.subject"] = subject
    pipeline = [{"$unwind": "$subjects"},
                {"$match": match},
                {"$project": {"_id": 0, "Student": "$student_id", "CGPA": "$prev_cgpa",
                              "Reappears": "$reappears", "Subject": "$subjects.subject",
                              "Attendance %": "$subjects.attendance_pct",
                              "CA1": "$subjects.ca1", "CA2": "$subjects.ca2", "CA3": "$subjects.ca3",
                              "Midterm": "$subjects.midterm", "End term": "$subjects.end_term",
                              "Total": "$subjects.total"}},
                {"$sort": {"Total": 1}}]
    rows = pd.DataFrame(list(students.aggregate(pipeline)))
    st.write(len(rows), "subject results found (lowest total first)")
    st.dataframe(rows, hide_index=True, width="stretch")


# ---------------------------------------------------------------
# Page 3: Student Lookup
# ---------------------------------------------------------------
elif page == "Student Lookup":
    st.title("Student Lookup")
    student_id = st.text_input("Student ID (S0001 to S1000)", "S0001").strip().upper()

    s = students.find_one({"student_id": student_id}, {"_id": 0})
    if s is None:
        st.error("No student found with ID " + student_id)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Previous CGPA", s["prev_cgpa"])
        col2.metric("Past reappears", s["reappears"])
        col3.metric("High Risk subjects", s.get("risk_subjects", 0))

        # For each subject: what the model predicts after the midterm vs what really happened
        table = []
        for sub in s["subjects"]:
            p = predict_subject(s["prev_cgpa"], s["reappears"], sub["subject"], sub["attendance_pct"],
                                sub["study_hours_daily"], sub["ca1"], sub["ca2"], sub["ca3"],
                                sub["midterm"])
            table.append({"Subject": sub["subject"], "Attendance %": sub["attendance_pct"],
                          "Study hrs/day": sub["study_hours_daily"],
                          "CA1": sub["ca1"], "CA2": sub["ca2"], "CA3": sub["ca3"],
                          "Midterm": sub["midterm"],
                          "Predicted end term": p["predicted_end_term"],
                          "Predicted risk": RISK_ICONS[p["risk_level"]],
                          "High Risk chance %": p["high_risk_probability"],
                          "Actual end term": sub["end_term"],
                          "Actual risk": RISK_ICONS[sub["risk_level"]]})
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


# ---------------------------------------------------------------
# Page 4: Predict New Student
# ---------------------------------------------------------------
elif page == "Predict New Student":
    st.title("Predict New Student")
    st.write("Enter the marks available right after the midterm.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        name = col1.text_input("Student name or ID", "New Student")
        prev_cgpa = col2.number_input("Previous semester CGPA", 0.0, 10.0, 7.0, 0.1)
        reappears = col3.number_input("Past reappears", 0, 10, 0)

        subject = st.selectbox("Subject", SUBJECTS)
        col1, col2 = st.columns(2)
        attendance = col1.slider("Attendance %", 0.0, 100.0, 80.0, 0.5)
        study_hours = col2.slider("Daily study hours", 0.0, 8.0, 2.5, 0.5)

        col1, col2, col3, col4 = st.columns(4)
        ca1 = col1.number_input("CA1 exam (/25)", 0, 25, 15)
        ca2 = col2.number_input("CA2 exam (/25)", 0, 25, 15)
        ca3 = col3.number_input("CA3 assignment (/25)", 0, 25, 17)
        midterm = col4.number_input("Midterm (/20)", 0, 20, 11)

        submitted = st.form_submit_button("Predict")

    if submitted:
        p = predict_subject(prev_cgpa, reappears, subject, attendance, study_hours,
                            ca1, ca2, ca3, midterm)

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted end term (/100)", p["predicted_end_term"])
        col2.metric("Predicted total (/100)", p["predicted_total"])
        col3.metric("High Risk chance", str(p["high_risk_probability"]) + "%")

        if p["risk_level"] == "High Risk":
            st.error("🔴 High Risk: likely to fail. Needs attention now.")
        elif p["risk_level"] == "Borderline":
            st.warning("🟡 Borderline: passing but weak. Keep an eye on this student.")
        else:
            st.success("🟢 Safe: on track to pass comfortably.")

        # Save the prediction in MongoDB so there is a record of every check
        predictions.insert_one({"name": name, "prev_cgpa": prev_cgpa, "reappears": int(reappears),
                                "subject": subject, "attendance_pct": attendance,
                                "study_hours_daily": study_hours, "ca1": int(ca1), "ca2": int(ca2),
                                "ca3": int(ca3), "midterm": int(midterm), **p,
                                "created_at": datetime.now()})
        st.caption("Saved to the 'predictions' collection in MongoDB.")

    # Show the latest saved predictions (newest first)
    st.subheader("Recent predictions")
    recent = list(predictions.find({}, {"_id": 0, "name": 1, "subject": 1, "predicted_end_term": 1,
                                        "risk_level": 1, "high_risk_probability": 1, "created_at": 1})
                  .sort("created_at", -1).limit(10))
    if recent:
        st.dataframe(pd.DataFrame(recent), hide_index=True, width="stretch")
    else:
        st.write("No predictions yet.")


# ---------------------------------------------------------------
# Page 5: Model Performance
# ---------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Tested on 200 students (1,000 subject results) the models never saw during training.")

    with open("outputs/metrics.json") as f:
        m = json.load(f)

    st.subheader("Linear Regression (end-term marks)")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", round(m["r2"], 2))
    col2.metric("MAE (marks)", round(m["mae"], 1))
    col3.metric("RMSE (marks)", round(m["rmse"], 1))

    st.subheader("Logistic Regression (High Risk or not)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", str(round(m["accuracy"] * 100, 1)) + "%")
    col2.metric("Precision", str(round(m["precision"] * 100, 1)) + "%")
    col3.metric("Recall", str(round(m["recall"] * 100, 1)) + "%")
    col4.metric("F1 score", round(m["f1"], 2))
    st.metric("Risk level accuracy (3 levels)", str(round(m["risk_level_accuracy"] * 100, 1)) + "%")

    st.subheader("Graphs")
    graphs = ["actual_vs_predicted", "prediction_errors", "confusion_matrix_logistic",
              "confusion_matrix_risk_levels", "risk_level_counts", "correlation_heatmap"]
    col1, col2 = st.columns(2)
    for i, g in enumerate(graphs):
        (col1 if i % 2 == 0 else col2).image("outputs/graphs/" + g + ".png")
    st.image("outputs/graphs/feature_weights.png")
