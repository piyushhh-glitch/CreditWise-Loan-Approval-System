import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Artifacts
# -----------------------------
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="💳",
    layout="centered"
)

st.title("💳 CreditWise Loan Approval System")
st.write(
    "Predict whether a loan application is likely to be approved using a Gaussian Naive Bayes model."
)

# -----------------------------
# Inputs
# -----------------------------
st.subheader("Applicant Details")

applicant_income = st.number_input(
    "Applicant Income", min_value=0.0, value=50000.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income", min_value=0.0, value=10000.0
)

age = st.number_input(
    "Age", min_value=18, max_value=100, value=30
)

dependents = st.number_input(
    "Dependents", min_value=0, max_value=10, value=0
)

existing_loans = st.number_input(
    "Existing Loans", min_value=0, max_value=20, value=0
)

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=900, value=750
)

dti_ratio = st.slider(
    "DTI Ratio",
    min_value=0.10,
    max_value=0.60,
    value=0.35,
    step=0.01
)

savings = st.number_input(
    "Savings", min_value=0.0, value=100000.0
)

collateral_value = st.number_input(
    "Collateral Value", min_value=0.0, value=500000.0
)

loan_amount = st.number_input(
    "Loan Amount", min_value=0.0, value=200000.0
)

loan_term = st.number_input(
    "Loan Term (Months)", min_value=1, value=60
)

education_level = st.selectbox(
    "Education Level",
    ["No", "Yes"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Contract", "Salaried", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Business", "Government", "MNC", "Private", "Unemployed"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Approval"):

    education_encoded = 1 if education_level == "Yes" else 0

    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2

    data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": education_encoded,
        "DTI_Ratio_sq": dti_ratio_sq,
        "Credit_Score_sq": credit_score_sq
    }

    # Initialize all remaining columns
    for col in feature_names:
        if col not in data:
            data[col] = 0

    # Employment Status
    if employment_status == "Salaried":
        data["Employment_Status_Salaried"] = 1
    elif employment_status == "Self-employed":
        data["Employment_Status_Self-employed"] = 1
    elif employment_status == "Unemployed":
        data["Employment_Status_Unemployed"] = 1

    # Marital Status
    if marital_status == "Single":
        data["Marital_Status_Single"] = 1

    # Loan Purpose
    if loan_purpose == "Car":
        data["Loan_Purpose_Car"] = 1
    elif loan_purpose == "Education":
        data["Loan_Purpose_Education"] = 1
    elif loan_purpose == "Home":
        data["Loan_Purpose_Home"] = 1
    elif loan_purpose == "Personal":
        data["Loan_Purpose_Personal"] = 1

    # Property Area
    if property_area == "Semiurban":
        data["Property_Area_Semiurban"] = 1
    elif property_area == "Urban":
        data["Property_Area_Urban"] = 1

    # Gender
    if gender == "Male":
        data["Gender_Male"] = 1

    # Employer Category
    if employer_category == "Government":
        data["Employer_Category_Government"] = 1
    elif employer_category == "MNC":
        data["Employer_Category_MNC"] = 1
    elif employer_category == "Private":
        data["Employer_Category_Private"] = 1
    elif employer_category == "Unemployed":
        data["Employer_Category_Unemployed"] = 1

    # DataFrame
    input_df = pd.DataFrame([data])

    # Match exact training feature order
    input_df = input_df[feature_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.metric(
            "Approval Probability",
            f"{probabilities[1] * 100:.2f}%"
        )
    else:
        st.error("❌ Loan Rejected")
        st.metric(
            "Rejection Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

    st.subheader("Model Confidence")
    st.write(
        f"Approval Probability: {probabilities[1] * 100:.2f}%"
    )
    st.write(
        f"Rejection Probability: {probabilities[0] * 100:.2f}%"
    )