# rules.py
# Our college's marking rules, written once and reused by every script
# (data generation, models and the Streamlit app).


def attendance_marks(attendance):
    """Convert attendance % into marks out of 5."""
    if attendance >= 90:
        return 5
    elif attendance >= 85:
        return 4
    elif attendance >= 80:
        return 3
    elif attendance >= 75:
        return 2
    elif attendance >= 70:
        return 1
    return 0


def ca_marks(ca1, ca2, ca3):
    """Best of the two CA exams plus the assignment (each out of 25), scaled to 25."""
    return (max(ca1, ca2) + ca3) / 2


def total_marks(attendance, ca1, ca2, ca3, midterm, end_term):
    """Total out of 100: attendance 5 + CA 25 + midterm 20 + end term (out of 100) scaled to 50."""
    return attendance_marks(attendance) + ca_marks(ca1, ca2, ca3) + midterm + end_term / 2


def risk_level(total, end_term):
    """Return the risk level for one subject.

    High Risk  -> likely to fail (total < 40 or end term < 30 out of 100)
    Borderline -> passing but weak (total 40 to 50)
    Safe       -> total above 50
    """
    if total < 40 or end_term < 30:
        return "High Risk"
    elif total <= 50:
        return "Borderline"
    return "Safe"
