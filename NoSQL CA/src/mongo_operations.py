# mongo_operations.py
# Phase 2: examples of MongoDB operations using PyMongo.
# Covers CRUD (Create, Read, Update, Delete), queries on embedded arrays,
# aggregation pipelines ($unwind, $group) and indexes on the 'students' collection.
# Run db.py first to load the data, then:  python src/mongo_operations.py

from db import get_collection

students = get_collection()


def title(text):
    """Print a heading so the output is easy to follow."""
    print("\n=== " + text + " ===")


# ---------------------------------------------------------------
# 1. READ: basic queries with find()
# ---------------------------------------------------------------
title("1. Total number of students")
print(students.count_documents({}))

title("2. One student's details (find_one with projection)")
# Projection picks which fields to show (0 = hide, 1 = show)
s = students.find_one({"student_id": "S0001"}, {"_id": 0})
print("ID:", s["student_id"], "| CGPA:", s["prev_cgpa"], "| reappears:", s["reappears"])
for sub in s["subjects"]:
    print("  ", sub["subject"], "| attendance:", sub["attendance_pct"],
          "| CA1/CA2/CA3:", sub["ca1"], sub["ca2"], sub["ca3"],
          "| mid:", sub["midterm"], "| end term:", sub["end_term"], "|", sub["risk_level"])

title("3. Students with CGPA below 6 AND 2 or more reappears")
# $lt = less than, $gte = greater than or equal. Two conditions = both must be true.
print(students.count_documents({"prev_cgpa": {"$lt": 6}, "reappears": {"$gte": 2}}), "students")

title("4. Top 5 students by previous CGPA")
# sort(-1) = highest first, limit(5) = first five only
for s in students.find({}, {"_id": 0, "student_id": 1, "prev_cgpa": 1}).sort("prev_cgpa", -1).limit(5):
    print(s)


# ---------------------------------------------------------------
# 2. QUERIES ON THE EMBEDDED 'subjects' ARRAY
# ---------------------------------------------------------------
title("5. Students at risk in at least one subject (dot notation)")
# "subjects.at_risk" looks inside every element of the array
print(students.count_documents({"subjects.at_risk": 1}), "students")

title("6. Students with attendance below 75% in Maths ($elemMatch)")
# $elemMatch makes BOTH conditions apply to the SAME subject in the array
query = {"subjects": {"$elemMatch": {"subject": "Maths", "attendance_pct": {"$lt": 75}}}}
print(students.count_documents(query), "students")

title("7. Show only the Maths marks of student S0002 ($elemMatch projection)")
s = students.find_one({"student_id": "S0002"},
                      {"_id": 0, "student_id": 1, "subjects": {"$elemMatch": {"subject": "Maths"}}})
print(s)


# ---------------------------------------------------------------
# 3. UPDATE
# ---------------------------------------------------------------
title("8. Count at-risk subjects for every student (update_many)")
# $sum over "$subjects.at_risk" adds up the 0/1 flags of the 5 subjects.
# The square brackets mean this is an "update with aggregation pipeline".
r = students.update_many({}, [{"$set": {"risk_subjects": {"$sum": "$subjects.at_risk"}}}])
print("Set the 'risk_subjects' field on", r.matched_count, "students")

title("9. Change one subject's mark with the positional $ operator (update_one)")
# "subjects.$" means "the array element that matched the filter" (here: DBMS)
query = {"student_id": "S0001", "subjects.subject": "DBMS"}
before = students.find_one(query, {"subjects.$": 1})["subjects"][0]["ca3"]
students.update_one(query, {"$inc": {"subjects.$.ca3": 2}})    # $inc adds 2 marks
after = students.find_one(query, {"subjects.$": 1})["subjects"][0]["ca3"]
print("S0001 DBMS CA3:", before, "->", after)
students.update_one(query, {"$inc": {"subjects.$.ca3": -2}})   # undo the demo change


# ---------------------------------------------------------------
# 4. CREATE and DELETE: add a new student, then remove them
# ---------------------------------------------------------------
title("10. Insert a new student (insert_one) and add a subject ($push)")
new_student = {"student_id": "S9999", "prev_cgpa": 8.1, "reappears": 0,
               "subjects": [{"subject": "DBMS", "attendance_pct": 92.0, "study_hours_daily": 3.0,
                             "ca1": 20, "ca2": 18, "ca3": 22, "midterm": 16}]}
students.insert_one(new_student)
# $push adds a new element to the end of an array
students.update_one({"student_id": "S9999"},
                    {"$push": {"subjects": {"subject": "Python", "attendance_pct": 88.0,
                                            "study_hours_daily": 2.5, "ca1": 19, "ca2": 21,
                                            "ca3": 23, "midterm": 15}}})
s = students.find_one({"student_id": "S9999"})
print("S9999 subjects:", [sub["subject"] for sub in s["subjects"]])

title("11. Delete that student (delete_one)")
result = students.delete_one({"student_id": "S9999"})
print("Deleted", result.deleted_count, "document. Total now:", students.count_documents({}))


# ---------------------------------------------------------------
# 5. AGGREGATION: summarise data inside MongoDB
# ---------------------------------------------------------------
title("12. Subject-wise summary ($unwind + $group)")
# $unwind turns 1 student with 5 subjects into 5 separate documents,
# then $group works like GROUP BY in SQL.
pipeline = [
    {"$unwind": "$subjects"},
    {"$group": {"_id": "$subjects.subject",
                "avg_attendance": {"$avg": "$subjects.attendance_pct"},
                "avg_midterm": {"$avg": "$subjects.midterm"},
                "avg_end_term": {"$avg": "$subjects.end_term"},
                "at_risk": {"$sum": "$subjects.at_risk"}}},
    {"$sort": {"at_risk": -1}},
]
for row in students.aggregate(pipeline):
    print(row["_id"].ljust(9), "| attendance:", round(row["avg_attendance"], 1),
          "| midterm:", round(row["avg_midterm"], 1),
          "| end term:", round(row["avg_end_term"], 1), "| at risk:", row["at_risk"])

title("13. Students at risk in 2 or more subjects")
# Way 1: $unwind, keep only at-risk subjects, count them per student, keep counts >= 2
pipeline = [
    {"$unwind": "$subjects"},
    {"$match": {"subjects.at_risk": 1}},
    {"$group": {"_id": "$student_id", "risky": {"$sum": 1}}},
    {"$match": {"risky": {"$gte": 2}}},
    {"$count": "students"},
]
print("Using aggregation:", list(students.aggregate(pipeline))[0]["students"], "students")
# Way 2: use the 'risk_subjects' field we added in step 8
print("Using find():     ", students.count_documents({"risk_subjects": {"$gte": 2}}), "students")

title("14. At-risk % by attendance marks (0 to 5)")
# $project calculates a new field: the percentage of at-risk subjects in each group
pipeline = [
    {"$unwind": "$subjects"},
    {"$group": {"_id": "$subjects.attendance_marks", "total": {"$sum": 1},
                "at_risk": {"$sum": "$subjects.at_risk"}}},
    {"$project": {"total": 1, "risk_pct": {"$multiply": [{"$divide": ["$at_risk", "$total"]}, 100]}}},
    {"$sort": {"_id": 1}},
]
for row in students.aggregate(pipeline):
    print("attendance marks", row["_id"], "| subjects:", row["total"],
          "| risk %:", round(row["risk_pct"], 1))

title("15. At-risk % by number of past reappears")
pipeline = [
    {"$unwind": "$subjects"},
    {"$group": {"_id": "$reappears", "total": {"$sum": 1},
                "at_risk": {"$sum": "$subjects.at_risk"}}},
    {"$project": {"total": 1, "risk_pct": {"$multiply": [{"$divide": ["$at_risk", "$total"]}, 100]}}},
    {"$sort": {"_id": 1}},
]
for row in students.aggregate(pipeline):
    print("reappears", row["_id"], "| subjects:", row["total"],
          "| risk %:", round(row["risk_pct"], 1))


title("16. How many subject results fall in each risk level")
pipeline = [
    {"$unwind": "$subjects"},
    {"$group": {"_id": "$subjects.risk_level", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]
for row in students.aggregate(pipeline):
    print(row["_id"].ljust(10), "|", row["count"], "subject results")


# ---------------------------------------------------------------
# 6. INDEXES: speed up frequent queries
# ---------------------------------------------------------------
title("17. Create indexes and list them")
students.create_index("prev_cgpa")
# An index on a field inside an array is called a "multikey" index
students.create_index([("subjects.subject", 1), ("subjects.at_risk", 1)])
for name in students.index_information():
    print(name)
