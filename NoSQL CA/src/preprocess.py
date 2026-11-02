# preprocess.py
# Phase 3: get the data out of MongoDB and prepare it for machine learning.
# Run db.py first, then:  python src/preprocess.py
#
# Steps:
#   1. Read from MongoDB (one row per student per subject, using $unwind)
#   2. Check data quality (missing values, duplicates, valid mark ranges)
#   3. Choose features: only what we KNOW right after the midterm
#   4. Encode the subject name as numbers (one-hot encoding)
#   5. Split into train and test sets (by student)
#   6. Scale the features (StandardScaler)
#   7. Save everything for Phase 4, plus a correlation heatmap

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")             # save graphs to files without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from db import get_collection

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)


# ---------------------------------------------------------------
# 1. Read from MongoDB
# ---------------------------------------------------------------
# Each student document has 5 subjects inside it. $unwind splits them so
# we get 5 rows per student, and $project picks and flattens the fields.
pipeline = [
    {"$unwind": "$subjects"},
    {"$project": {
        "_id": 0,
        "student_id": 1,
        "prev_cgpa": 1,
        "reappears": 1,
        "subject": "$subjects.subject",
        "attendance_pct": "$subjects.attendance_pct",
        "study_hours_daily": "$subjects.study_hours_daily",
        "ca1": "$subjects.ca1",
        "ca2": "$subjects.ca2",
        "ca3": "$subjects.ca3",
        "midterm": "$subjects.midterm",
        "end_term": "$subjects.end_term",
        "risk_level": "$subjects.risk_level",
        "at_risk": "$subjects.at_risk",
    }},
]
df = pd.DataFrame(list(get_collection().aggregate(pipeline)))
print("Rows read from MongoDB:", df.shape[0], "| columns:", df.shape[1])
print(df.head(), "\n")


# ---------------------------------------------------------------
# 2. Data quality checks
# ---------------------------------------------------------------
print("Missing values in each column:")
print(df.isnull().sum().to_string(), "\n")

print("Duplicate rows:", df.duplicated().sum())

# Marks must be inside their allowed range, otherwise the data is wrong
ranges = {"attendance_pct": (0, 100), "ca1": (0, 25), "ca2": (0, 25), "ca3": (0, 25),
          "midterm": (0, 20), "end_term": (0, 100), "prev_cgpa": (0, 10)}
for col, (low, high) in ranges.items():
    bad = ((df[col] < low) | (df[col] > high)).sum()
    print("Out-of-range values in", col.ljust(15), ":", bad)

# If anything is missing, drop those rows (our generated data has none)
df = df.dropna()

print("\nRisk level counts:")
print(df["risk_level"].value_counts().to_string(), "\n")


# ---------------------------------------------------------------
# 3. Choose features (inputs) and targets (outputs)
# ---------------------------------------------------------------
# We predict right after the midterm, so we can only use what is known by then.
# end_term, risk_level and at_risk are the answers we want to predict,
# so they must NOT be used as inputs.
FEATURES = ["prev_cgpa", "reappears", "attendance_pct", "study_hours_daily",
            "ca1", "ca2", "ca3", "midterm"]

# Correlation heatmap: how strongly each input moves with the end-term mark
plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES + ["end_term"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between inputs and end-term marks")
plt.tight_layout()
plt.savefig("outputs/graphs/correlation_heatmap.png")
plt.close()
print("Saved outputs/graphs/correlation_heatmap.png")


# ---------------------------------------------------------------
# 4. One-hot encode the subject
# ---------------------------------------------------------------
# Models only understand numbers, so "Maths" becomes a column subject_Maths = 1/0.
# drop_first=True drops one subject (DBMS) to avoid repeating information:
# if all the other subject columns are 0, the subject must be DBMS.
df = pd.get_dummies(df, columns=["subject"], drop_first=True, dtype=int)
subject_cols = [c for c in df.columns if c.startswith("subject_")]
FEATURES = FEATURES + subject_cols
print("Final input features:", FEATURES, "\n")


# ---------------------------------------------------------------
# 5. Train / test split (80% / 20%) by student
# ---------------------------------------------------------------
# All 5 subjects of one student go into the SAME set, so the model is
# tested on students it has never seen before (a fair test).
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["student_id"]))
train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
print("Train rows:", len(train), "(" + str(train["student_id"].nunique()) + " students)")
print("Test rows: ", len(test), "(" + str(test["student_id"].nunique()) + " students)")
print("High Risk share -> train:", round(train["at_risk"].mean() * 100, 1), "% | test:",
      round(test["at_risk"].mean() * 100, 1), "%\n")


# ---------------------------------------------------------------
# 6. Scale the features
# ---------------------------------------------------------------
# Features have very different ranges (CGPA 4-10, attendance 40-100).
# StandardScaler changes each one to mean 0 and standard deviation 1,
# so no feature dominates just because its numbers are bigger.
# The scaler learns ONLY from the training data, then is applied to both sets.
scaler = StandardScaler()
train[FEATURES] = scaler.fit_transform(train[FEATURES])
test[FEATURES] = scaler.transform(test[FEATURES])


# ---------------------------------------------------------------
# 7. Save for Phase 4
# ---------------------------------------------------------------
keep = ["student_id"] + FEATURES + ["end_term", "risk_level", "at_risk"]
train[keep].to_csv("data/processed/train.csv", index=False)
test[keep].to_csv("data/processed/test.csv", index=False)
joblib.dump(scaler, "models/scaler.pkl")          # needed later to scale new students in the app
joblib.dump(FEATURES, "models/features.pkl")      # the exact column order the models expect
print("Saved data/processed/train.csv, data/processed/test.csv,")
print("      models/scaler.pkl and models/features.pkl")
