import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Job Prediction System", page_icon="🎓")

st.markdown("""
<style>

.stApp{
background-color:#0A2A4A;
}

h1,h2,h3,h4,h5,h6{
color:white !important;
}

label{
color:white !important;
}

div.stButton > button{
background-color:#1E90FF;
color:white;
border-radius:8px;
border:none;
padding:10px 20px;
font-weight:bold;
}

div.stButton > button:hover{
background-color:#1565C0;
}

/* Dropdown selected text */
.stSelectbox div[data-baseweb="select"] > div{
color:black !important;
}

/* Dropdown options text */
div[data-baseweb="popover"] *{
color:black !important;
}

/* Sidebar */
section[data-testid="stSidebar"] *{
color:black !important;
}

/* Prediction box */
.prediction-box{
background-color:#0f3d5c;
padding:15px;
border-radius:10px;
font-size:20px;
font-weight:bold;
color:#ffffff;
}

</style>
""", unsafe_allow_html=True)


# Load dataset
df = pd.read_csv("job_dataset.csv")

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
degree_encoder = pickle.load(open("degree_encoder.pkl", "rb"))
spec_encoder = pickle.load(open("spec_encoder.pkl", "rb"))
job_encoder = pickle.load(open("job_encoder.pkl", "rb"))


# Create users.csv if not exists
if not os.path.exists("users.csv"):
    pd.DataFrame(columns=["username","password"]).to_csv("users.csv", index=False)


# Session states
if "login" not in st.session_state:
    st.session_state.login = False

if "page" not in st.session_state:
    st.session_state.page = "Register"


# ---------------- REGISTER PAGE ---------------- #

def register():
    st.title("EDU JOB Prediction System")
    st.title("📝 Register")

    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Register"):

        users = pd.read_csv("users.csv")

        username = username.strip()
        password = password.strip()

        users["username"] = users["username"].astype(str).str.strip()

        if username in users["username"].values:
            st.warning("User already exists")

        else:
            new_user = pd.DataFrame([[username,password]], columns=["username","password"])
            users = pd.concat([users,new_user], ignore_index=True)
            users.to_csv("users.csv", index=False)

            st.success("Registration successful!")
            st.session_state.page = "Login"
            st.rerun()


# ---------------- LOGIN PAGE ---------------- #

def login():
    st.title("EDU JOB Prediction System")
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        users = pd.read_csv("users.csv")

        username = username.strip()
        password = password.strip()

        users["username"] = users["username"].astype(str).str.strip()
        users["password"] = users["password"].astype(str).str.strip()

        if username in users["username"].values:

            stored_password = users.loc[users["username"] == username, "password"].values[0]

            if stored_password == password:
                st.session_state.login = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()

            else:
                st.error("Incorrect password")

        else:
            st.error("User not found")


# ---------------- DASHBOARD ---------------- #

def dashboard():

    col1, col2 = st.columns([6,2])

    with col1:
        st.title("🎓 Job Role Prediction System")

    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.login = False
            st.session_state.page = "Login"
            st.rerun()
    st.subheader(f"Welcome, {st.session_state.username} 👋")
    st.subheader("Academic Details")

    col1, col2 = st.columns(2)

    with col1:
        degree = st.selectbox("Degree", df["Degree"].unique())

    with col2:
        specialization = st.selectbox("Specialization", df["Specialization"].unique())

    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)


    # -------- NEW INPUTS -------- #

    skills = st.text_input(
        "Skills (Enter multiple skills separated by commas)",
        placeholder="Python, Java, SQL, Machine Learning"
    )

    certificates = st.text_input(
        "Certificates (Enter multiple certificates separated by commas)",
        placeholder="AWS, Coursera ML, Google Data Analytics"
    )


    # -------- PREDICTION -------- #

    if st.button("Predict Job Role"):

        degree_encoded = degree_encoder.transform([degree])[0]
        spec_encoded = spec_encoder.transform([specialization])[0]

        input_data = [[degree_encoded, spec_encoded, cgpa]]

        prediction = model.predict(input_data)

        job = job_encoder.inverse_transform(prediction)

        st.markdown(
            f"""
            <div class="prediction-box">
            Predicted Job Role: {job[0]}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ---------------- DATASET INSIGHTS ---------------- #

    st.subheader("📊 Dataset Insights")

    st.write("Dataset Preview")
    st.dataframe(df)

    job_counts = df["JobRole"].value_counts()

    st.write("Job Role Distribution")
    st.bar_chart(job_counts)

    st.write("CGPA Distribution")
    st.line_chart(df["CGPA"])


# ---------------- MAIN ---------------- #

if st.session_state.login:
    dashboard()

else:

    menu = st.sidebar.selectbox(
        "Menu",
        ["Register","Login"],
        index = 0 if st.session_state.page == "Register" else 1
    )

    if menu == "Register":
        register()
    else:
        login()