import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("loan_default_model.pkl")

st.title("🏦 Loan Default Prediction System")
st.markdown("Predict whether a loan will be defaulted based on applicant data.")

# Form Inputs
with st.form("loan_form"):
    age = st.slider("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 1000, 1000000, 25000)
    loan_amount = st.number_input("Loan Amount", 1000, 1000000, 50000)
    term = st.selectbox("Loan Term (Months)", [12, 24, 36, 60])
    credit_score = st.slider("Credit Score", 300, 850, 650)
    dependents = st.number_input("No. of Dependents", 0, 10, 1)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    submit = st.form_submit_button("Predict")

    if submit:
        # Encode categorical
        education_encoded = 1 if education == "Graduate" else 0
        marital_encoded = 1 if marital_status == "Married" else 0

        # Input array
        input_data = np.array([[age, income, loan_amount, term, credit_score,
                                dependents, education_encoded, marital_encoded]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 High risk of default. Probability: {probability:.2%}")
        else:
            st.success(f"✅ Low risk of default. Probability: {probability:.2%}")
