# db.py
# Phase 2: connecting to MongoDB and loading the dataset into it.
# Other scripts import get_collection() so the connection details live in one place.
# Run generate_data.py first, then:  python src/db.py

import json
from pymongo import MongoClient

# Connection settings (local MongoDB server, default port)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "student_ews"          # ews = Early Warning System
COLLECTION_NAME = "students"
JSON_PATH = "data/students.json"


def get_collection():
    """Connect to MongoDB and return the 'students' collection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]          # database is created automatically on first insert
    return db[COLLECTION_NAME]


def load_json_to_mongo():
    """Insert every student as one MongoDB document.

    Each document looks like:
    {
      "student_id": "S0001", "prev_cgpa": 7.4, "reappears": 0,
      "subjects": [ {"subject": "DBMS", "attendance_pct": 88.0, "ca1": 17, ...}, ... ]
    }
    The 5 subjects are stored INSIDE the student document as an embedded array.
    This is a typical NoSQL design: everything about one student is read in one go,
    with no joins needed (in SQL this would be two tables: students and marks).
    """
    with open(JSON_PATH) as f:
        records = json.load(f)

    students = get_collection()
    students.drop()                          # start fresh (removes old data and indexes)
    result = students.insert_many(records)   # bulk insert all students at once

    # Store how many subjects each student is High Risk in (used by the app)
    students.update_many({}, [{"$set": {"risk_subjects": {"$sum": "$subjects.at_risk"}}}])

    # Index on student_id makes lookups by id fast and stops duplicate ids
    students.create_index("student_id", unique=True)

    print("Inserted", len(result.inserted_ids), "students into",
          DB_NAME + "." + COLLECTION_NAME)


if __name__ == "__main__":
    load_json_to_mongo()
