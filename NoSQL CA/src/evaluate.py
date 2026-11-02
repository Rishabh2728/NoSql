# evaluate.py
# Phase 5: test both models on the 200 students they have NEVER seen
# and draw graphs for the report.
# Run train_models.py first, then:  python src/evaluate.py
#
#   Linear Regression   -> R², MAE, RMSE
#   Logistic Regression -> Accuracy, Precision, Recall, F1, confusion matrix
#   Risk levels         -> how often the predicted level (High Risk / Borderline / Safe)
#                          matches the real one

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # save graphs to files without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from rules import total_marks, risk_level

GRAPHS = "outputs/graphs/"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
RISK_COLORS = {"High Risk": "#d03b3b", "Borderline": "#fab219", "Safe": "#0ca30c"}
LEVELS = ["High Risk", "Borderline", "Safe"]

# Simple, clean look for every graph
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titleweight": "bold", "font.size": 11})

# Load the test data and the saved models
FEATURES = joblib.load("models/features.pkl")
scaler = joblib.load("models/scaler.pkl")
lin_model = joblib.load("models/linear_regression.pkl")
log_model = joblib.load("models/logistic_regression.pkl")
test = pd.read_csv("data/processed/test.csv")
X_test = test[FEATURES]


# ---------------------------------------------------------------
# 1. Linear Regression: how close are the predicted marks?
# ---------------------------------------------------------------
y_true = test["end_term"]
y_pred = np.clip(lin_model.predict(X_test), 0, 100)   # marks can't go below 0 or above 100

r2 = r2_score(y_true, y_pred)                      # 1.0 = perfect, 0 = no better than the average
mae = mean_absolute_error(y_true, y_pred)          # average size of the error, in marks
rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # like MAE but punishes big errors more

print("=== Linear Regression (end-term marks) ===")
print("R²   :", round(r2, 3))
print("MAE  :", round(mae, 2), "marks")
print("RMSE :", round(rmse, 2), "marks")


# ---------------------------------------------------------------
# 2. Logistic Regression: how well does it spot High Risk students?
# ---------------------------------------------------------------
risk_true = test["at_risk"]
risk_pred = log_model.predict(X_test)

accuracy = accuracy_score(risk_true, risk_pred)    # % of all predictions that are correct
precision = precision_score(risk_true, risk_pred)  # of those we flagged, % that really were High Risk
recall = recall_score(risk_true, risk_pred)        # of the real High Risk, % that we caught
f1 = f1_score(risk_true, risk_pred)                # balance of precision and recall
cm = confusion_matrix(risk_true, risk_pred)

print("\n=== Logistic Regression (High Risk or not) ===")
print("Accuracy  :", round(accuracy, 3))
print("Precision :", round(precision, 3))
print("Recall    :", round(recall, 3))
print("F1 score  :", round(f1, 3))
print("Confusion matrix [rows = actual, columns = predicted]:")
print(cm)


# ---------------------------------------------------------------
# 3. Risk levels from the predicted marks
# ---------------------------------------------------------------
# The test features are scaled, so turn them back into real marks first
real = pd.DataFrame(scaler.inverse_transform(X_test), columns=FEATURES)
pred_levels = []
for i in range(len(test)):
    r = real.iloc[i]
    total = total_marks(r["attendance_pct"], r["ca1"], r["ca2"], r["ca3"], r["midterm"], y_pred[i])
    pred_levels.append(risk_level(total, y_pred[i]))
level_accuracy = accuracy_score(test["risk_level"], pred_levels)
level_cm = confusion_matrix(test["risk_level"], pred_levels, labels=LEVELS)

print("\n=== Risk level (from predicted marks + college rules) ===")
print("Accuracy:", round(level_accuracy, 3))
print("Confusion matrix [rows = actual, columns = predicted]  order:", LEVELS)
print(level_cm)

# Save all numbers so the report and the app can use them
metrics = {"r2": r2, "mae": mae, "rmse": rmse,
           "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
           "risk_level_accuracy": level_accuracy}
metrics = {name: round(float(value), 4) for name, value in metrics.items()}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved outputs/metrics.json")


# ---------------------------------------------------------------
# 4. Graphs
# ---------------------------------------------------------------
# Graph 1: actual vs predicted end-term marks
plt.figure(figsize=(7, 6))
plt.scatter(y_true, y_pred, s=14, alpha=0.5, color=BLUE)
plt.plot([0, 100], [0, 100], color=GRAY, linestyle="--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual end-term mark")
plt.ylabel("Predicted end-term mark")
plt.title("Linear Regression: actual vs predicted (R² = " + str(round(r2, 2)) + ")")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(GRAPHS + "actual_vs_predicted.png", dpi=120)
plt.close()

# Graph 2: how big the errors are
plt.figure(figsize=(7, 5))
plt.hist(y_pred - y_true, bins=30, color=BLUE, edgecolor="white")
plt.axvline(0, color=GRAY, linestyle="--", linewidth=1.5)
plt.xlabel("Prediction error (predicted − actual), in marks")
plt.ylabel("Number of subject results")
plt.title("Linear Regression: prediction errors (MAE = " + str(round(mae, 1)) + " marks)")
plt.tight_layout()
plt.savefig(GRAPHS + "prediction_errors.png", dpi=120)
plt.close()

# Graph 3: confusion matrix for Logistic Regression
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=2, linecolor="white",
            xticklabels=["Not High Risk", "High Risk"], yticklabels=["Not High Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression: confusion matrix")
plt.tight_layout()
plt.savefig(GRAPHS + "confusion_matrix_logistic.png", dpi=120)
plt.close()

# Graph 4: confusion matrix for the 3 risk levels
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(level_cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=2,
            linecolor="white", xticklabels=LEVELS, yticklabels=LEVELS)
plt.xlabel("Predicted level")
plt.ylabel("Actual level")
plt.title("Risk levels: actual vs predicted (accuracy " + str(round(level_accuracy * 100)) + "%)")
plt.tight_layout()
plt.savefig(GRAPHS + "confusion_matrix_risk_levels.png", dpi=120)
plt.close()

# Graph 5: which inputs matter most (weights of both models)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, coefs, name in [(axes[0], lin_model.coef_, "Linear Regression\n(effect on end-term mark)"),
                        (axes[1], log_model.coef_[0], "Logistic Regression\n(effect on chance of High Risk)")]:
    w = pd.Series(coefs, index=FEATURES).sort_values()
    colors = [ORANGE if v < 0 else BLUE for v in w]
    ax.barh(w.index, w.values, color=colors, height=0.6)
    ax.axvline(0, color="#c3c2b7", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("Weight (blue = increases, orange = decreases)")
plt.tight_layout()
plt.savefig(GRAPHS + "feature_weights.png", dpi=120)
plt.close()

# Graph 6: how many subject results are in each risk level (actual vs predicted)
actual_counts = test["risk_level"].value_counts().reindex(LEVELS)
pred_counts = pd.Series(pred_levels).value_counts().reindex(LEVELS, fill_value=0)
x = np.arange(len(LEVELS))
plt.figure(figsize=(7, 5))
plt.bar(x - 0.2, actual_counts, width=0.38, color=[RISK_COLORS[l] for l in LEVELS], label="Actual")
plt.bar(x + 0.2, pred_counts, width=0.38, color=[RISK_COLORS[l] for l in LEVELS],
        alpha=0.45, hatch="//", edgecolor="white", label="Predicted")
for i in range(len(LEVELS)):
    plt.text(x[i] - 0.2, actual_counts.iloc[i] + 8, str(actual_counts.iloc[i]), ha="center", color="#52514e")
    plt.text(x[i] + 0.2, pred_counts.iloc[i] + 8, str(pred_counts.iloc[i]), ha="center", color="#52514e")
plt.xticks(x, LEVELS)
plt.ylabel("Number of subject results (test set)")
plt.title("Risk levels: actual vs predicted")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(GRAPHS + "risk_level_counts.png", dpi=120)
plt.close()

print("Saved 6 graphs in", GRAPHS)
