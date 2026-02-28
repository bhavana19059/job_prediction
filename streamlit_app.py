import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ----------------------------
# Load Dataset for Model
# ----------------------------
df = pd.read_csv("job_dataset.csv")

le_degree = LabelEncoder()
le_specialization = LabelEncoder()
le_job = LabelEncoder()

df['Degree'] = le_degree.fit_transform(df['Degree'])
df['Specialization'] = le_specialization.fit_transform(df['Specialization'])
df['JobRole'] = le_job.fit_transform(df['JobRole'])

X = df[['Degree', 'Specialization', 'CGPA']]
y = df['JobRole']

model = DecisionTreeClassifier()
model.fit(X, y)

# ----------------------------
# Session State Setup
# ----------------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if "page" not in st.session_state:
    st.session_state.page = "signup"

# ----------------------------
# SIGNUP PAGE
# ----------------------------
if st.session_state.page == "signup":
    st.title("Create Account")

    name = st.text_input("Enter Name")
    email = st.text_input("Enter Email")
    password = st.text_input("Enter Password", type="password")

    if st.button("Register"):
        st.session_state.users[email] = password
        st.success("Account Created Successfully!")
        st.session_state.page = "login"

    if st.button("Go to Login"):
        st.session_state.page = "login"

# ----------------------------
# LOGIN PAGE
# ----------------------------
elif st.session_state.page == "login":
    st.title("Login Page")

    email = st.text_input("Enter Email")
    password = st.text_input("Enter Password", type="password")

    if st.button("Login"):
        if email in st.session_state.users and st.session_state.users[email] == password:
            st.success("Login Successful!")
            st.session_state.page = "predict"
        else:
            st.error("Invalid Credentials")

# ----------------------------
# PREDICTION PAGE
# ----------------------------
elif st.session_state.page == "predict":
    st.title("🎓 Job Prediction System")

    degree = st.selectbox("Select Degree", le_degree.classes_)
    specialization = st.selectbox("Select Specialization", le_specialization.classes_)
    cgpa = st.number_input("Enter CGPA", 0.0, 10.0)

    if st.button("Predict Job"):
        d = le_degree.transform([degree])[0]
        s = le_specialization.transform([specialization])[0]

        prediction = model.predict([[d, s, cgpa]])
        job = le_job.inverse_transform(prediction)[0]

        st.success(f"Predicted Job Role: {job}")

    # ----------------------------
    # Job Role Distribution Graph (REAL NAMES)
    # ----------------------------
    st.subheader("Job Role Distribution")

    df_display = pd.read_csv("job_dataset.csv")
    st.bar_chart(df_display['JobRole'].value_counts())

    if st.button("Logout"):
        st.session_state.page = "login"