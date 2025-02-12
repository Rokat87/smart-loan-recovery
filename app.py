import streamlit as st
import joblib
import pandas as pd
import os
import urllib.request

# Download model and preprocessing artifacts if they don't exist
def download_file(url, file_path):
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

MODEL_URL = "https://github.com/yourusername/smart-loan-recovery/releases/download/v1.0/loan_recovery_model.pkl"
SCALER_URL = "https://github.com/yourusername/smart-loan-recovery/releases/download/v1.0/scaler.pkl"
LABEL_ENCODERS_URL = "https://github.com/yourusername/smart-loan-recovery/releases/download/v1.0/label_encoders.pkl"

download_file(MODEL_URL, "loan_recovery_model.pkl")
download_file(SCALER_URL, "scaler.pkl")
download_file(LABEL_ENCODERS_URL, "label_encoders.pkl")

# Load preprocessing artifacts and model
model = joblib.load("loan_recovery_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Rest of your app code...
# Load necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
from streamlit_authenticator import Authenticate
import yaml

# Load preprocessing artifacts and model
model = joblib.load('loan_recovery_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define a function to preprocess user inputs
def preprocess_input(age, gender, employment_type, monthly_income, loan_amount, loan_tenure, interest_rate, 
                     loan_type, collateral_value, outstanding_loan_amount, monthly_emi, num_missed_payments, days_past_due):
    # Encode categorical inputs
    gender = label_encoders['Gender'].transform([gender])[0]
    employment_type = label_encoders['Employment_Type'].transform([employment_type])[0]
    loan_type = label_encoders['Loan_Type'].transform([loan_type])[0]

    # Normalize numeric inputs
    input_data = np.array([
        age, gender, employment_type, monthly_income, loan_amount, loan_tenure, interest_rate,
        collateral_value, outstanding_loan_amount, monthly_emi, num_missed_payments, days_past_due
    ]).reshape(1, -1)
    input_data = scaler.transform(input_data)

    return input_data

# Streamlit app
st.title("Smart Loan Recovery System")

# User authentication
with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.success(f"Logged in as {name}")
else:
    st.warning("Please log in to use the system.")
    st.stop()

# Input section
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ['Male', 'Female'])
employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business Owner'])
monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
loan_tenure = st.number_input("Loan Tenure (Months)", min_value=1, value=60)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0, step=0.1)
loan_type = st.selectbox("Loan Type", ['Home', 'Auto', 'Personal', 'Business'])
collateral_value = st.number_input("Collateral Value", min_value=0, value=0)
outstanding_loan_amount = st.number_input("Outstanding Loan Amount", min_value=0, value=0)
monthly_emi = st.number_input("Monthly EMI", min_value=0, value=0)
num_missed_payments = st.number_input("Number of Missed Payments", min_value=0, value=0)
days_past_due = st.number_input("Days Past Due", min_value=0, value=0)

# Prediction section
if st.button("Predict"):
    input_data = preprocess_input(
        age, gender, employment_type, monthly_income, loan_amount, loan_tenure, interest_rate,
        loan_type, collateral_value, outstanding_loan_amount, monthly_emi, num_missed_payments, days_past_due
    )
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    recovery_status = label_encoders['Recovery_Status'].inverse_transform([prediction])[0]
    st.subheader("Prediction Results")
    st.write(f"**Recovery Status:** {recovery_status}")
    st.write(f"**Probability Scores:**")
    st.write(f"- Fully Recovered: {probabilities[0]:.2f}")
    st.write(f"- Partially Recovered: {probabilities[1]:.2f}")
    st.write(f"- Written Off: {probabilities[2]:.2f}")

# Batch processing
st.header("Batch Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Preprocess batch data
    for col in categorical_cols:
        le = label_encoders[col]
        batch_data[col] = le.transform(batch_data[col].astype(str))

    batch_data[numeric_cols] = scaler.transform(batch_data[numeric_cols])

    # Generate predictions
    batch_predictions = model.predict(batch_data[X.columns])
    batch_probabilities = model.predict_proba(batch_data[X.columns])

    # Add predictions and probabilities to the DataFrame
    batch_data['Recovery_Status'] = label_encoders['Recovery_Status'].inverse_transform(batch_predictions)
    batch_data['Fully_Recovered_Prob'] = [p[0] for p in batch_probabilities]
    batch_data['Partially_Recovered_Prob'] = [p[1] for p in batch_probabilities]
    batch_data['Written_Off_Prob'] = [p[2] for p in batch_probabilities]

    # Display results
    st.subheader("Batch Prediction Results")
    st.dataframe(batch_data)

# Visualizations
st.header("Visualizations")
if 'Recovery_Status' in batch_data.columns:
    fig = px.histogram(batch_data, x='Recovery_Status', title="Distribution of Recovery Status")
    st.plotly_chart(fig)

# Risk scoring
def calculate_risk_score(num_missed_payments, days_past_due):
    weight_missed_payments = 0.6
    weight_days_past_due = 0.4
    normalized_missed_payments = min(num_missed_payments / 12, 1)  # Cap at 12 months
    normalized_days_past_due = min(days_past_due / 90, 1)  # Cap at 90 days
    risk_score = (weight_missed_payments * normalized_missed_payments) + (weight_days_past_due * normalized_days_past_due)
    return round(risk_score * 100, 2)

if st.button("Calculate Risk Score"):
    risk_score = calculate_risk_score(num_missed_payments, days_past_due)
    st.write(f"**Risk Score:** {risk_score}%")
