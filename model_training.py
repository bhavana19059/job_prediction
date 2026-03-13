import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- LOAD DATASET ---------------- #

df = pd.read_csv("job_dataset.csv")

# ---------------- ENCODERS ---------------- #

degree_encoder = LabelEncoder()
spec_encoder = LabelEncoder()
job_encoder = LabelEncoder()

df["Degree"] = degree_encoder.fit_transform(df["Degree"])
df["Specialization"] = spec_encoder.fit_transform(df["Specialization"])
df["JobRole"] = job_encoder.fit_transform(df["JobRole"])

# ---------------- FEATURES & TARGET ---------------- #

X = df[["Degree","Specialization","CGPA"]]
y = df["JobRole"]

# ---------------- MODEL ---------------- #

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# ---------------- SAVE MODEL ---------------- #

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(degree_encoder, open("degree_encoder.pkl","wb"))
pickle.dump(spec_encoder, open("spec_encoder.pkl","wb"))
pickle.dump(job_encoder, open("job_encoder.pkl","wb"))

print("Model training completed successfully!")
print("Files created:")
print("model.pkl")
print("degree_encoder.pkl")
print("spec_encoder.pkl")
print("job_encoder.pkl")