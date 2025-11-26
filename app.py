import streamlit as st
import pandas as pd
import joblib

# Load models once (cached)
@st.cache_resource
def load_models():
    rf = joblib.load("loan_model_rf.pkl")
    log = joblib.load("loan_model_log.pkl")
    return rf, log

rf_model, log_model = load_models()

# UI Setup
st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")
st.title("Loan Eligibility Prediction System")
st.write("Select a model and enter applicant details to get prediction.")

# Select model
model_choice = st.radio(
    "Choose a prediction model:",
    ["Random Forest", "Logistic Regression"]
)

# Input Form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0,1,2,3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    applicant_income = st.number_input("Applicant Income", min_value=0, value=3000)
    co_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amt = st.number_input("Loan Amount", min_value=0, value=150)
    loan_term = st.number_input("Loan Term", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Eligibility")

if submitted:
    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amt,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    df_input = pd.DataFrame([input_data])

    # Choose model
    model = rf_model if model_choice == "Random Forest" else log_model

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    status = "Eligible" if pred == 1 else "Not Eligible"

    st.subheader("Result")
    st.write(f"**Prediction:** {status}")
    st.write(f"**Approval Probability:** {proba:.2%}")

    st.json(input_data)
