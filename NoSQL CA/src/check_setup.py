# check_setup.py
# Phase 1: quick check that everything needed for the project is ready.
# Run it from the project folder:  python src/check_setup.py

import json
from pymongo import MongoClient

# 1. Check the dataset exists (create it with: python src/generate_data.py)
with open("data/students.json") as f:
    data = json.load(f)
print("Dataset loaded:", len(data), "students,", len(data[0]["subjects"]), "subjects each")

# 2. Check we can talk to the local MongoDB server
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")  # raises an error if MongoDB is not running
    print("MongoDB is running, version", client.server_info()["version"])
except Exception as e:
    print("Could not connect to MongoDB:", e)
