# generate_data.py
# Creates a synthetic dataset that follows our college's marking scheme.
# Output: data/students.json  (1,000 students, 5 subjects each)
# Run it from the project folder:  python src/generate_data.py
#
# Marking scheme (total = 100 per subject):
#   Attendance  ->  5 marks  (90%+ = 5, 85%+ = 4, 80%+ = 3, 75%+ = 2, 70%+ = 1, below = 0)
#   CA          -> 25 marks  (best of CA1/CA2 exam + CA3 assignment, each out of 25, averaged)
#   Midterm     -> 20 marks
#   End term    -> 50 marks  (exam is out of 100, scaled down to 50)
# Risk levels for each subject:
#   High Risk  -> likely to fail (total < 40 or end-term exam < 30 out of 100)
#   Borderline -> passing but weak (total 40 to 50)
#   Safe       -> total above 50
#
# How the data is made realistic:
#   Every student gets a hidden "ability" score. Better students tend to have
#   higher CGPA, attend more, study more and score more. Random noise is added
#   everywhere so the pattern is believable but not perfect.

import json
import numpy as np
from rules import attendance_marks, ca_marks, risk_level   # college marking rules

NUM_STUDENTS = 1000
SUBJECTS = ["DBMS", "Python", "Maths", "OS", "Networks"]
# Some subjects are harder than others (negative = harder)
SUBJECT_DIFFICULTY = {"DBMS": 0.1, "Python": 0.3, "Maths": -0.4, "OS": -0.1, "Networks": 0.0}

rng = np.random.default_rng(42)   # fixed seed -> same data every run


def clip(value, low, high):
    """Keep a value inside a range, e.g. marks between 0 and 25."""
    return float(min(max(value, low), high))


students = []
for i in range(1, NUM_STUDENTS + 1):
    ability = rng.normal(0, 1)    # hidden student ability (average student = 0)

    # Student-level fields
    prev_cgpa = round(clip(7 + 1.1 * ability + rng.normal(0, 0.5), 4, 10), 2)
    # Weaker students are more likely to have reappears
    reappears = int(rng.poisson(max(0.05, 0.6 - 0.5 * ability)))

    subjects = []
    for name in SUBJECTS:
        # How well this student does in this subject
        skill = ability + SUBJECT_DIFFICULTY[name] + rng.normal(0, 0.6)

        attendance = round(clip(83 + 5 * skill + rng.normal(0, 7), 40, 100), 1)
        study_hours = round(clip(2.5 + 0.8 * skill + rng.normal(0, 0.8), 0, 8) * 2) / 2  # nearest 0.5
        ca1 = round(clip(15 + 4 * skill + rng.normal(0, 3), 0, 25))
        ca2 = round(clip(15 + 4 * skill + rng.normal(0, 3), 0, 25))
        ca3 = round(clip(17 + 3 * skill + rng.normal(0, 2.5), 0, 25))   # assignments score a bit higher
        midterm = round(clip(11 + 3.5 * skill + rng.normal(0, 2.5), 0, 20))

        # End-term exam (out of 100) depends on the earlier marks and habits, plus noise
        end_term = round(clip(
            20
            + 1.0 * ca_marks(ca1, ca2, ca3)
            + 1.4 * midterm
            + 0.15 * (attendance - 75)
            + 2.0 * study_hours
            + 2.5 * (prev_cgpa - 7)
            - 3.0 * reappears
            + rng.normal(0, 10),
            0, 100))

        # Final result using the college rules
        att_m = attendance_marks(attendance)
        ca_m = ca_marks(ca1, ca2, ca3)
        total = round(att_m + ca_m + midterm + end_term / 2, 1)
        level = risk_level(total, end_term)
        at_risk = 1 if level == "High Risk" else 0

        subjects.append({
            "subject": name,
            "attendance_pct": attendance,
            "study_hours_daily": study_hours,
            "ca1": ca1, "ca2": ca2, "ca3": ca3,
            "midterm": midterm,
            "attendance_marks": att_m,
            "ca_marks": ca_m,
            "end_term": end_term,     # target for Linear Regression
            "total": total,
            "risk_level": level,      # High Risk / Borderline / Safe
            "at_risk": at_risk,       # 1 = High Risk, target for Logistic Regression
        })

    students.append({
        "student_id": "S" + str(i).zfill(4),   # S0001, S0002, ...
        "prev_cgpa": prev_cgpa,
        "reappears": reappears,
        "subjects": subjects,
    })

with open("data/students.json", "w") as f:
    json.dump(students, f, indent=2)

# Quick summary so we can check the data looks sensible
rows = [s for st in students for s in st["subjects"]]
print("Students:", len(students), "| subject records:", len(rows))
print("At-risk subject records:", sum(r["at_risk"] for r in rows),
      "(" + str(round(100 * sum(r["at_risk"] for r in rows) / len(rows), 1)) + "%)")
for lvl in ["High Risk", "Borderline", "Safe"]:
    print(" ", lvl + ":", sum(r["risk_level"] == lvl for r in rows))
print("Average end-term mark:", round(np.mean([r["end_term"] for r in rows]), 1))
