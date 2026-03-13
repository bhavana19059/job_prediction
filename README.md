# Job Prediction Using Educational Details

## Project Description

This project predicts a suitable job role based on a user's educational details such as degree, specialization, and CGPA. The system uses a machine learning model to analyze the input and suggest a possible job role.

The project also includes a simple web application created using Streamlit where users can enter their educational details and get a predicted job role.

## Technologies Used

* Python
* Machine Learning
* Pandas
* Scikit-learn
* Streamlit

## Project Files

model_training.py
Used to train the machine learning model using the dataset.

streamlit_app.py
Used to create the web application interface for job prediction.

job_dataset.csv
Dataset used to train the machine learning model.

model.pkl
Saved machine learning model used for prediction.

degree_encoder.pkl
Encoder used for degree values.

spec_encoder.pkl
Encoder used for specialization values.

job_encoder.pkl
Encoder used for job labels.

users.csv
File used to store user details.

## How to Run the Project

1. Install required libraries

pip install pandas scikit-learn streamlit

2. Run the Streamlit application

streamlit run streamlit_app.py

3. Open the browser link shown in the terminal to use the application.
