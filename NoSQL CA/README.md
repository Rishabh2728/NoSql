# Early Warning System for At-Risk Students

MongoDB + PyMongo + scikit-learn project. Predicts, right after the midterm,
which students are likely to fail a subject, using our college's marking scheme.

## Marking scheme (per subject, out of 100)

| Component  | Marks | Rule |
|------------|-------|------|
| Attendance | 5     | 90%+ = 5, 85%+ = 4, 80%+ = 3, 75%+ = 2, 70%+ = 1, below = 0 |
| CA         | 25    | (best of CA1/CA2 exam + CA3 assignment) / 2 |
| Midterm    | 20    | |
| End term   | 50    | exam out of 100, scaled to 50 |

Risk levels per subject:
- High Risk: likely to fail (total < 40 or end-term exam < 30/100)
- Borderline: passing but weak (total 40 to 50)
- Safe: total above 50

The rules live in `src/rules.py`.

## Dataset

Synthetic, made by `src/generate_data.py`: 1,000 students, 5 subjects each
(DBMS, Python, Maths, OS, Networks). Student fields: previous CGPA, reappears.
Subject fields: attendance %, daily study hours, CA1, CA2, CA3, midterm, end term.

## Folder structure

```
data/            students.json (generated dataset)
src/             Python scripts (data, database, preprocessing, models)
models/          saved trained models
outputs/graphs/  evaluation charts
report/          final CA report
```

## How to run

```
pip install -r requirements.txt
python src/generate_data.py     # create data/students.json
python src/db.py                # load it into MongoDB
python src/mongo_operations.py  # CRUD, queries and aggregation examples
python src/preprocess.py        # clean, encode, split and scale for the models
python src/train_models.py      # train Linear and Logistic Regression
python src/predict.py           # example predictions for new students
python src/evaluate.py          # metrics (outputs/metrics.json) and graphs (outputs/graphs/)
streamlit run app.py            # open the web app
```
