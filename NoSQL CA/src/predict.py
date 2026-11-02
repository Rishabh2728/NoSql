# predict.py
# Phase 4: use the trained models to predict the result of a NEW student.
# The Streamlit app (Phase 6) will call predict_subject().
# Try it from the project folder:  python src/predict.py

import joblib
import pandas as pd
from rules import total_marks, risk_level

SUBJECTS = ["DBMS", "Python", "Maths", "OS", "Networks"]

# Load everything saved in Phases 3 and 4
scaler = joblib.load("models/scaler.pkl")
FEATURES = joblib.load("models/features.pkl")
lin_model = joblib.load("models/linear_regression.pkl")
log_model = joblib.load("models/logistic_regression.pkl")


def predict_subject(prev_cgpa, reappears, subject, attendance_pct, study_hours_daily,
                    ca1, ca2, ca3, midterm):
    """Predict one subject for one student, right after the midterm."""
    # 1. Put the inputs in a one-row table, in the same format as the training data
    row = {"prev_cgpa": prev_cgpa, "reappears": reappears, "attendance_pct": attendance_pct,
           "study_hours_daily": study_hours_daily, "ca1": ca1, "ca2": ca2, "ca3": ca3,
           "midterm": midterm}
    for s in SUBJECTS[1:]:                    # one-hot subject columns (DBMS = all zeros)
        row["subject_" + s] = 1 if subject == s else 0
    X = pd.DataFrame([row])[FEATURES]

    # 2. Scale it with the SAME scaler used in training
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURES)

    # 3. Linear Regression -> predicted end-term mark (kept between 0 and 100)
    end_term = float(lin_model.predict(X_scaled)[0])
    end_term = min(max(end_term, 0), 100)

    # 4. Apply the college rules to the predicted mark -> total and risk level
    total = total_marks(attendance_pct, ca1, ca2, ca3, midterm, end_term)
    level = risk_level(total, end_term)

    # 5. Logistic Regression -> probability of being High Risk
    fail_prob = float(log_model.predict_proba(X_scaled)[0][1])

    return {"subject": subject,
            "predicted_end_term": round(end_term, 1),
            "predicted_total": round(total, 1),
            "risk_level": level,
            "high_risk_probability": round(fail_prob * 100, 1)}


if __name__ == "__main__":
    # Example: a weak student and a strong student in Maths
    print(predict_subject(prev_cgpa=5.6, reappears=2, subject="Maths", attendance_pct=68,
                          study_hours_daily=1.0, ca1=9, ca2=11, ca3=13, midterm=7))
    print(predict_subject(prev_cgpa=8.4, reappears=0, subject="Maths", attendance_pct=92,
                          study_hours_daily=3.5, ca1=20, ca2=19, ca3=21, midterm=16))
