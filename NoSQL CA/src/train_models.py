# train_models.py
# Phase 4: train the two machine learning models.
# Run preprocess.py first, then:  python src/train_models.py
#
#   Model 1: Linear Regression   -> predicts the END-TERM mark (0 to 100)
#   Model 2: Logistic Regression -> predicts if the subject is HIGH RISK (1) or not (0)
#
# Both models use only what is known right after the midterm.

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

FEATURES = joblib.load("models/features.pkl")    # same column order as in Phase 3
train = pd.read_csv("data/processed/train.csv")

X_train = train[FEATURES]            # inputs (already scaled in Phase 3)
y_marks = train["end_term"]          # target for Linear Regression
y_risk = train["at_risk"]            # target for Logistic Regression


# ---------------------------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------------------------
# Finds the straight-line formula:
#   end_term = intercept + w1*prev_cgpa + w2*reappears + ... + w12*subject_Python
# choosing the weights (w) that make the smallest errors on the training data.
lin_model = LinearRegression()
lin_model.fit(X_train, y_marks)

print("=== Linear Regression: weight of each input ===")
print("(inputs are scaled, so a bigger weight = a bigger effect on the end-term mark)")
weights = pd.Series(lin_model.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
print(weights.round(2).to_string())
print("Intercept (average end-term mark):", round(lin_model.intercept_, 2))


# ---------------------------------------------------------------
# Model 2: Logistic Regression
# ---------------------------------------------------------------
# Works like Linear Regression, but passes the result through a sigmoid curve
# so the output is a probability between 0 and 1 (chance of being High Risk).
# If the probability is 0.5 or more, the subject is predicted as High Risk.
#
# Only about 15% of results are High Risk. class_weight="balanced" tells the
# model that missing a High Risk student is a bigger mistake than a false alarm,
# which is what we want from an early warning system.
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train, y_risk)

print("\n=== Logistic Regression: weight of each input ===")
print("(positive = increases the chance of High Risk, negative = decreases it)")
weights = pd.Series(log_model.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
print(weights.round(2).to_string())


# ---------------------------------------------------------------
# Save both models for Phase 5 (evaluation) and Phase 6 (app)
# ---------------------------------------------------------------
joblib.dump(lin_model, "models/linear_regression.pkl")
joblib.dump(log_model, "models/logistic_regression.pkl")
print("\nSaved models/linear_regression.pkl and models/logistic_regression.pkl")
