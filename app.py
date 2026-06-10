import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — SecureTrust Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #0a0d14 !important;
    color: #e2e8f0;
}

/* ── HERO BAND ── */
.hero {
    background: linear-gradient(135deg, #0f1829 0%, #0a1628 50%, #0f1420 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 2.8rem 3rem 2.2rem;
    margin: -1rem -1rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: #f8fafc;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-title span { color: #60a5fa; }
.hero-sub {
    color: #64748b;
    font-size: 0.92rem;
    max-width: 520px;
    line-height: 1.6;
}
.hero-stat-row {
    display: flex;
    gap: 2rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.hero-stat .hs-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #f1f5f9;
}
.hero-stat .hs-lbl {
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 0.8rem;
    vertical-align: middle;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1829;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #475569;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #fff !important;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.8rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e3a5f;
}

/* ── INPUT CARD ── */
.input-card {
    background: #0f1829;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.input-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

/* ── SCORE RING (applicant health) ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem 1rem;
}

/* ── RESULT PANELS ── */
.result-approved {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
}
.result-rejected {
    background: linear-gradient(135deg, #1a0505 0%, #450a0a 100%);
    border: 1px solid #dc2626;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.result-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.result-sub { font-size: 0.88rem; opacity: 0.75; }

/* ── REJECTION REASON CARD ── */
.reason-card {
    background: #1a0505;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #dc2626;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.reason-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #fca5a5;
    margin-bottom: 0.25rem;
}
.reason-body {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.55;
}

/* ── IMPROVE CARD ── */
.improve-card {
    background: #0c1f3a;
    border: 1px solid #1d4ed8;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.improve-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 0.25rem;
}
.improve-body {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.55;
}

/* ── METRIC TILES ── */
.tile {
    background: #0f1829;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.tile-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
}
.tile-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #f1f5f9;
}
.tile-note { font-size: 0.72rem; color: #64748b; }

/* ── RISK METER ── */
.risk-bar-wrap { margin: 0.4rem 0 1rem; }
.risk-bar-bg {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] {
    background: #0f1829 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
}
div[data-testid="stMetricLabel"] p { color: #64748b !important; font-size: 0.78rem !important; }
div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #4f46e5 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

label[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div { background: #0f1829 !important; border-color: #1e3a5f !important; }
div[data-baseweb="input"] > div { background: #0f1829 !important; border-color: #1e3a5f !important; }

.stDataFrame { border: 1px solid #1e3a5f !important; border-radius: 10px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN PIPELINE (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising CreditWise engine…")
def build_pipeline():
    df = pd.read_csv("dataset/loan_approval_data.csv")

    categ_cols = df.select_dtypes(include=["object"]).columns
    num_cols   = df.select_dtypes(include=["float64", "int64"]).columns

    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categ_cols] = cat_imp.fit_transform(df[categ_cols])

    raw_df = df.copy()      # pre-encoding, for EDA

    df = df.drop(columns=["Applicant_ID"])
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])
    # LabelEncoder: No→0, Yes→1

    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_arr  = ohe.fit_transform(df[ohe_cols])
    enc_df   = pd.DataFrame(enc_arr, columns=ohe.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), enc_df], axis=1)

    df["DTI_Ratio_sq"]    = df["DTI_Ratio"]   ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved","DTI_Ratio","Credit_Score"])
    y = df["Loan_Approved"]
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(Xtr_s, y_train)
    yp = model.predict(Xte_s)

    perf = {
        "accuracy":  accuracy_score(y_test, yp),
        "precision": precision_score(y_test, yp),
        "recall":    recall_score(y_test, yp),
        "f1":        f1_score(y_test, yp),
        "cm":        confusion_matrix(y_test, yp),
    }

    return model, scaler, ohe, feature_cols, raw_df, df, perf


model, scaler, ohe, feature_cols, raw_df, enc_df, perf = build_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# REJECTION ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
def analyse_rejection(credit_score, dti_ratio, applicant_income,
                      coapplicant_income, savings, loan_amount,
                      existing_loans, collateral_value, loan_term,
                      employment_status, age):
    reasons   = []
    tips      = []
    total_inc = applicant_income + coapplicant_income
    lti       = loan_amount / total_inc if total_inc > 0 else 99
    monthly   = loan_amount / loan_term if loan_term > 0 else 0

    # Credit score
    if credit_score < 620:
        reasons.append(("Low Credit Score",
            f"Your score of {credit_score:.0f} is below the preferred threshold of 650. "
            "Lenders use this as the primary risk indicator."))
        tips.append(("Improve Credit Score",
            "Pay all existing EMIs on time for 6+ months. Reduce credit card utilisation below 30%. "
            "Check your credit report for errors and dispute any inaccuracies."))
    elif credit_score < 660:
        reasons.append(("Below-Average Credit Score",
            f"Your score of {credit_score:.0f} is in the borderline zone (620–659). "
            "Most approvals happen above 660."))
        tips.append(("Boost Credit Score",
            "Even a 30-40 point improvement could change the outcome. "
            "Avoid new credit applications for 3–6 months and clear small overdue balances first."))

    # DTI
    if dti_ratio > 0.50:
        reasons.append(("High Debt-to-Income Ratio",
            f"Your DTI of {dti_ratio:.0%} means over half your income is already committed to debt. "
            "Lenders typically cap this at 40–50%."))
        tips.append(("Reduce Existing Debt",
            f"Close or pay down existing loans to bring DTI below 40%. "
            "Paying off one loan could reduce your DTI significantly."))
    elif dti_ratio > 0.40:
        reasons.append(("Elevated DTI Ratio",
            f"DTI of {dti_ratio:.0%} is above the comfortable 40% ceiling."))
        tips.append(("Lower DTI",
            "Consider prepaying one existing loan before reapplying. "
            "Even small reductions in outstanding debt improve your DTI."))

    # Loan-to-Income
    if lti > 5:
        reasons.append(("Loan Amount Too High Relative to Income",
            f"Requested ₹{loan_amount:,.0f} is {lti:.1f}× your combined income. "
            "Banks typically approve up to 4–5× annual income."))
        tips.append(("Request a Lower Amount",
            f"Try requesting ₹{total_inc * 4:,.0f} or less. "
            "Alternatively, including a co-applicant with income can raise your eligibility."))

    # Savings / Collateral
    if savings < loan_amount * 0.10:
        reasons.append(("Insufficient Savings",
            f"Savings of ₹{savings:,.0f} are less than 10% of the loan amount. "
            "Lenders prefer applicants to have a financial buffer."))
        tips.append(("Build Savings",
            "Accumulate at least 10–15% of the loan amount in liquid savings. "
            "This signals repayment capacity and reduces lender risk."))

    if collateral_value < loan_amount * 0.50 and loan_amount > 100000:
        reasons.append(("Low Collateral Coverage",
            f"Collateral value ₹{collateral_value:,.0f} covers only "
            f"{collateral_value/loan_amount:.0%} of the loan. "
            "For large loans, lenders prefer 60–80% coverage."))
        tips.append(("Provide Additional Collateral",
            "Adding property, gold, or FD as collateral improves your security ratio and reassures the lender."))

    # Existing loans
    if existing_loans >= 3:
        reasons.append(("Too Many Existing Loans",
            f"You already have {int(existing_loans)} active loans. "
            "Lenders view this as an over-leveraged profile."))
        tips.append(("Clear Existing Loans",
            "Closing 1–2 existing loans before reapplying will significantly improve your profile. "
            "Start with the highest-interest ones."))

    # Employment
    if employment_status == "Unemployed":
        reasons.append(("No Active Employment",
            "Unemployed applicants cannot demonstrate stable repayment capacity, "
            "which is a primary requirement for loan approval."))
        tips.append(("Secure Employment First",
            "Wait until you have at least 6 months of stable employment income before reapplying. "
            "Even a part-time income declaration helps."))

    # Age edge cases
    if age < 22:
        reasons.append(("Limited Credit History (Young Applicant)",
            f"At {int(age)}, you likely have a thin credit file. "
            "Banks prefer applicants with at least 2 years of credit history."))
        tips.append(("Build Credit History",
            "Start with a secured credit card or a small personal loan repaid on time. "
            "12–18 months of good repayment history is usually enough."))

    if not reasons:
        reasons.append(("Marginal Profile — Multiple Small Risk Factors",
            "No single dominant red flag was found, but the combination of your income, "
            "debt, and credit profile falls just below the approval threshold."))
        tips.append(("Small Improvements Across the Board",
            "Improve credit score by 20–30 points, reduce DTI by 5%, "
            "and add ₹20,000–₹50,000 to savings before reapplying."))

    return reasons, tips


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
approved_pct = (raw_df["Loan_Approved"] == "Yes").mean() * 100
total_rec     = len(raw_df)

st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">SecureTrust Bank · AI Credit Division</div>
  <div class="hero-title">Credit<span>Wise</span> <span class="hero-badge">Naive Bayes</span></div>
  <div class="hero-sub">Intelligent, unbiased loan assessment powered by machine learning.
    Instant decisions — with full transparency on why.</div>
  <div class="hero-stat-row">
    <div class="hero-stat">
      <span class="hs-val">{perf['accuracy']:.1%}</span>
      <span class="hs-lbl">Model Accuracy</span>
    </div>
    <div class="hero-stat">
      <span class="hs-val">{perf['precision']:.1%}</span>
      <span class="hs-lbl">Precision</span>
    </div>
    <div class="hero-stat">
      <span class="hs-val">{perf['f1']:.1%}</span>
      <span class="hs-lbl">F1 Score</span>
    </div>
    <div class="hero-stat">
      <span class="hs-val">{total_rec:,}</span>
      <span class="hs-lbl">Training Records</span>
    </div>
    <div class="hero-stat">
      <span class="hs-val">{approved_pct:.0f}%</span>
      <span class="hs-lbl">Historical Approval Rate</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_model = st.tabs([
    "🔍  Loan Assessment",
    "📊  Data Insights",
    "🤖  Model Report",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════
with tab_predict:

    left_col, right_col = st.columns([3, 2], gap="large")

    # ── INPUT FORM ──────────────────────────────────────────────────────────
    with left_col:

        # Personal
        st.markdown('<div class="sec-label">Personal Information</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            age = st.number_input("Age", 18, 75, 32)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with p2:
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            dependents = st.number_input("Dependents", 0, 10, 1)
        with p3:
            education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        # Financial
        st.markdown('<div class="sec-label">Financial Profile</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            applicant_income = st.number_input("Applicant Income (₹/mo)", 0, 500000, 10000, step=500)
            coapplicant_income = st.number_input("Co-applicant Income (₹/mo)", 0, 500000, 3000, step=500)
        with f2:
            savings = st.number_input("Savings (₹)", 0, 2000000, 15000, step=1000)
            collateral_value = st.number_input("Collateral Value (₹)", 0, 5000000, 50000, step=5000)
        with f3:
            existing_loans = st.number_input("Existing Loans", 0, 10, 1)
            dti_ratio = st.slider("DTI Ratio", 0.10, 0.60, 0.35, 0.01,
                                  help="Debt-to-Income ratio — fraction of income going to debt payments")

        # Credit + Employment
        st.markdown('<div class="sec-label">Credit & Employment</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            credit_score = st.slider("Credit Score", 550, 800, 680,
                                     help="Credit bureau score (CIBIL / Experian)")
            employment_status = st.selectbox("Employment Status",
                ["Salaried", "Self-employed", "Contract", "Unemployed"])
        with e2:
            employer_category = st.selectbox("Employer Category",
                ["Private", "Government", "MNC", "Business", "Unemployed"])
        with e3:
            loan_purpose = st.selectbox("Loan Purpose",
                ["Home", "Car", "Education", "Personal", "Business"])

        # Loan
        st.markdown('<div class="sec-label">Loan Request</div>', unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        with l1:
            loan_amount = st.number_input("Loan Amount (₹)", 10000, 5000000, 150000, step=5000)
        with l2:
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84, 120, 180, 240, 360])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run Loan Assessment")

    # ── LIVE RISK PANEL ──────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="sec-label">Applicant Risk Dashboard</div>', unsafe_allow_html=True)

        total_income = applicant_income + coapplicant_income
        lti_ratio = round(loan_amount / total_income, 2) if total_income > 0 else 99
        monthly_emi = round(loan_amount / loan_term, 0) if loan_term > 0 else 0
        emi_to_income = monthly_emi / total_income if total_income > 0 else 1

        # Credit band
        if credit_score >= 750:   cs_label, cs_color = "Excellent", "#4ade80"
        elif credit_score >= 700: cs_label, cs_color = "Good",      "#86efac"
        elif credit_score >= 650: cs_label, cs_color = "Fair",      "#fbbf24"
        elif credit_score >= 600: cs_label, cs_color = "Poor",      "#f97316"
        else:                     cs_label, cs_color = "Very Poor", "#ef4444"

        # DTI band
        if dti_ratio < 0.30:   dti_label, dti_color = "Healthy",  "#4ade80"
        elif dti_ratio < 0.40: dti_label, dti_color = "Moderate", "#fbbf24"
        elif dti_ratio < 0.50: dti_label, dti_color = "High",     "#f97316"
        else:                  dti_label, dti_color = "Critical", "#ef4444"

        # LTI band
        if lti_ratio < 3:      lti_label, lti_color = "Conservative", "#4ade80"
        elif lti_ratio < 5:    lti_label, lti_color = "Moderate",     "#fbbf24"
        else:                  lti_label, lti_color = "Aggressive",   "#ef4444"

        # Compute composite risk score 0-100
        risk = 0
        risk += max(0, (700 - credit_score) / 1.5)        # credit score
        risk += min(30, dti_ratio * 50)                    # DTI
        risk += min(20, (lti_ratio - 1) * 4)              # LTI
        risk += existing_loans * 3                         # loans
        risk += (0 if employment_status == "Salaried" else 8)
        risk = min(100, max(0, risk))

        if risk < 30:    risk_label, risk_c = "Low Risk",    "#4ade80"
        elif risk < 55:  risk_label, risk_c = "Medium Risk", "#fbbf24"
        elif risk < 75:  risk_label, risk_c = "High Risk",   "#f97316"
        else:            risk_label, risk_c = "Very High",   "#ef4444"

        # Gauge chart
        fig_g, ax_g = plt.subplots(figsize=(4.5, 2.6))
        fig_g.patch.set_facecolor("#0f1829")
        ax_g.set_facecolor("#0f1829")
        theta = np.linspace(np.pi, 0, 200)
        ax_g.plot(np.cos(theta), np.sin(theta), color="#1e3a5f", linewidth=14, solid_capstyle="round")
        fill_end = np.pi - (risk / 100) * np.pi
        theta_f = np.linspace(np.pi, fill_end, 200)
        ax_g.plot(np.cos(theta_f), np.sin(theta_f), color=risk_c,
                  linewidth=14, solid_capstyle="round")
        ax_g.text(0, -0.15, f"{risk:.0f}", ha="center", va="center",
                  fontsize=28, fontweight="bold", color="#f1f5f9",
                  fontfamily="DM Sans")
        ax_g.text(0, -0.45, risk_label, ha="center", va="center",
                  fontsize=10, color=risk_c)
        ax_g.text(0, -0.68, "Composite Risk Score", ha="center", va="center",
                  fontsize=7.5, color="#475569")
        ax_g.set_xlim(-1.4, 1.4)
        ax_g.set_ylim(-0.85, 1.1)
        ax_g.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        # Quick metric tiles
        t1, t2 = st.columns(2)
        t1.markdown(f"""
        <div class="tile">
          <div class="tile-label">Credit Score</div>
          <div class="tile-value" style="color:{cs_color}">{credit_score}</div>
          <div class="tile-note">{cs_label}</div>
        </div>""", unsafe_allow_html=True)
        t2.markdown(f"""
        <div class="tile">
          <div class="tile-label">DTI Ratio</div>
          <div class="tile-value" style="color:{dti_color}">{dti_ratio:.0%}</div>
          <div class="tile-note">{dti_label}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t3, t4 = st.columns(2)
        t3.markdown(f"""
        <div class="tile">
          <div class="tile-label">Loan / Income</div>
          <div class="tile-value" style="color:{lti_color}">{lti_ratio:.1f}×</div>
          <div class="tile-note">{lti_label}</div>
        </div>""", unsafe_allow_html=True)
        t4.markdown(f"""
        <div class="tile">
          <div class="tile-label">Monthly EMI</div>
          <div class="tile-value">₹{monthly_emi:,.0f}</div>
          <div class="tile-note">{emi_to_income:.0%} of income</div>
        </div>""", unsafe_allow_html=True)

    # ── PREDICTION RESULT ───────────────────────────────────────────────────
    if predict_btn:
        edu_map = {"Graduate": 1, "Not Graduate": 0}
        edu_encoded = edu_map.get(education_level, 0)

        ohe_input = pd.DataFrame(
            [[employment_status, marital_status, loan_purpose,
              property_area, gender, employer_category]],
            columns=["Employment_Status","Marital_Status","Loan_Purpose",
                     "Property_Area","Gender","Employer_Category"]
        )
        ohe_encoded = ohe.transform(ohe_input)
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out())

        base = pd.DataFrame([{
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age":                age,
            "Dependents":         dependents,
            "Existing_Loans":     existing_loans,
            "Savings":            savings,
            "Collateral_Value":   collateral_value,
            "Loan_Amount":        loan_amount,
            "Loan_Term":          loan_term,
            "Education_Level":    edu_encoded,
        }])

        full = pd.concat([base.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        full["DTI_Ratio_sq"]    = dti_ratio ** 2
        full["Credit_Score_sq"] = credit_score ** 2

        for col in feature_cols:
            if col not in full.columns:
                full[col] = 0
        full = full[feature_cols]

        scaled = scaler.transform(full)
        pred   = model.predict(scaled)[0]          # 0=No, 1=Yes
        proba  = model.predict_proba(scaled)[0]    # [P(No), P(Yes)]

        st.markdown("---")
        st.markdown('<div class="sec-label">Assessment Result</div>', unsafe_allow_html=True)

        res_col, prob_col = st.columns([1, 1], gap="large")

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-approved">
                  <div class="result-icon">✅</div>
                  <div class="result-verdict" style="color:#4ade80">Loan Approved</div>
                  <div class="result-sub" style="color:#a7f3d0">
                    This applicant meets SecureTrust Bank's criteria.<br>
                    Approval confidence: <strong>{proba[1]:.1%}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                  <div class="result-icon">❌</div>
                  <div class="result-verdict" style="color:#f87171">Loan Rejected</div>
                  <div class="result-sub" style="color:#fca5a5">
                    This application does not meet the required criteria.<br>
                    Rejection confidence: <strong>{proba[0]:.1%}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)

        with prob_col:
            fig_p, ax_p = plt.subplots(figsize=(4, 2.8))
            fig_p.patch.set_facecolor("#0f1829")
            ax_p.set_facecolor("#0f1829")
            bars = ax_p.barh(["❌ Rejected", "✅ Approved"],
                             [proba[0], proba[1]],
                             color=["#dc2626","#10b981"],
                             height=0.45, edgecolor="#0a0d14")
            ax_p.set_xlim(0, 1.15)
            ax_p.set_xlabel("Probability", color="#475569", fontsize=9)
            ax_p.tick_params(colors="#94a3b8", labelsize=9)
            for spine in ax_p.spines.values():
                spine.set_edgecolor("#1e3a5f")
            for bar, val in zip(bars, [proba[0], proba[1]]):
                ax_p.text(val + 0.03, bar.get_y() + bar.get_height()/2,
                          f"{val:.1%}", va="center", color="white",
                          fontsize=10, fontweight="bold")
            ax_p.set_title("Decision Probability", color="#94a3b8", fontsize=9, pad=8)
            ax_p.set_facecolor("#0f1829")
            fig_p.patch.set_facecolor("#0f1829")
            plt.tight_layout()
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

        # ── REJECTION BREAKDOWN ─────────────────────────────────────────────
        if pred == 0:
            reasons, tips = analyse_rejection(
                credit_score, dti_ratio, applicant_income,
                coapplicant_income, savings, loan_amount,
                existing_loans, collateral_value, loan_term,
                employment_status, age
            )

            st.markdown("<br>", unsafe_allow_html=True)
            r_col, i_col = st.columns(2, gap="large")

            with r_col:
                st.markdown('<div class="sec-label">Why was this rejected?</div>', unsafe_allow_html=True)
                for title, body in reasons:
                    st.markdown(f"""
                    <div class="reason-card">
                      <div class="reason-title">⚠ {title}</div>
                      <div class="reason-body">{body}</div>
                    </div>""", unsafe_allow_html=True)

            with i_col:
                st.markdown('<div class="sec-label">How to improve eligibility</div>', unsafe_allow_html=True)
                for title, body in tips:
                    st.markdown(f"""
                    <div class="improve-card">
                      <div class="improve-title">💡 {title}</div>
                      <div class="improve-body">{body}</div>
                    </div>""", unsafe_allow_html=True)

        # ── FEATURE CONTRIBUTION CHART (NB log-likelihood delta) ───────────
        st.markdown('<div class="sec-label">Feature Impact on This Decision</div>', unsafe_allow_html=True)

        scaled_arr = scaled[0]
        feat_names_short = {
            "Applicant_Income": "Applicant Income",
            "Coapplicant_Income": "Co-applicant Income",
            "Age": "Age",
            "Dependents": "Dependents",
            "Existing_Loans": "Existing Loans",
            "Savings": "Savings",
            "Collateral_Value": "Collateral",
            "Loan_Amount": "Loan Amount",
            "Loan_Term": "Loan Term",
            "Education_Level": "Education",
            "DTI_Ratio_sq": "DTI² (Risk)",
            "Credit_Score_sq": "Credit² (Quality)",
        }

        # NB log-likelihood contribution per feature
        log_p1 = -0.5 * ((scaled_arr - model.theta_[1]) ** 2) / model.var_[1]
        log_p0 = -0.5 * ((scaled_arr - model.theta_[0]) ** 2) / model.var_[0]
        delta   = log_p1 - log_p0   # positive → pushes toward approval

        delta_df = pd.DataFrame({"feature": feature_cols, "delta": delta})
        delta_df["label"] = delta_df["feature"].map(
            lambda x: feat_names_short.get(x, x.replace("_"," ").title())
        )
        delta_df = delta_df.reindex(delta_df["delta"].abs().nlargest(12).index)
        delta_df = delta_df.sort_values("delta")

        fig_f, ax_f = plt.subplots(figsize=(8, 4))
        fig_f.patch.set_facecolor("#0f1829")
        ax_f.set_facecolor("#0f1829")
        colors = ["#10b981" if v > 0 else "#ef4444" for v in delta_df["delta"]]
        ax_f.barh(delta_df["label"], delta_df["delta"], color=colors, edgecolor="#0a0d14", height=0.6)
        ax_f.axvline(0, color="#334155", linewidth=1)
        ax_f.set_xlabel("← Pushes Rejection  |  Pushes Approval →",
                        color="#475569", fontsize=8.5)
        ax_f.tick_params(colors="#94a3b8", labelsize=9)
        for spine in ax_f.spines.values():
            spine.set_edgecolor("#1e3a5f")
        ax_f.set_title("Top 12 Features — Log-Likelihood Contribution (Naive Bayes)",
                       color="#94a3b8", fontsize=9, pad=10)
        plt.tight_layout()
        st.pyplot(fig_f, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda:
    def dark_fig(w=5.5, h=4):
        f, a = plt.subplots(figsize=(w, h))
        f.patch.set_facecolor("#0f1829")
        a.set_facecolor("#0f1829")
        for sp in a.spines.values():
            sp.set_edgecolor("#1e3a5f")
        a.tick_params(colors="#94a3b8", labelsize=9)
        return f, a

    st.markdown('<div class="sec-label">Dataset Summary</div>', unsafe_allow_html=True)
    ds1, ds2, ds3, ds4, ds5 = st.columns(5)
    yes_count = (raw_df["Loan_Approved"] == "Yes").sum()
    no_count  = (raw_df["Loan_Approved"] == "No").sum()
    ds1.metric("Total Applications", f"{len(raw_df):,}")
    ds2.metric("Approved", f"{yes_count:,}")
    ds3.metric("Rejected", f"{no_count:,}")
    ds4.metric("Approval Rate", f"{yes_count/len(raw_df):.1%}")
    ds5.metric("Features Used", "27")

    # Row 1
    st.markdown('<div class="sec-label">Approval & Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        f, a = dark_fig()
        counts = raw_df["Loan_Approved"].value_counts()
        wedges, texts, auto = a.pie(
            counts, labels=["Approved","Rejected"],
            colors=["#10b981","#ef4444"],
            autopct="%1.1f%%",
            textprops={"color":"white","fontsize":9},
            wedgeprops={"edgecolor":"#0a0d14","linewidth":2},
            startangle=90
        )
        for at in auto: at.set_fontweight("bold")
        a.set_title("Approval Split", color="#e2e8f0", fontsize=11, pad=10)
        st.pyplot(f, use_container_width=True); plt.close()

    with col2:
        f, a = dark_fig()
        emp = raw_df["Employment_Status"].value_counts()
        palette = ["#3b82f6","#6366f1","#8b5cf6","#a78bfa","#c4b5fd"][:len(emp)]
        bars = a.barh(emp.index, emp.values, color=palette, edgecolor="#0a0d14", height=0.55)
        a.bar_label(bars, padding=4, color="#e2e8f0", fontsize=8)
        a.set_xlabel("Applicants", color="#475569", fontsize=9)
        a.set_title("Employment Status", color="#e2e8f0", fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    with col3:
        f, a = dark_fig()
        prop = raw_df["Property_Area"].value_counts()
        a.bar(prop.index, prop.values,
              color=["#0ea5e9","#0284c7","#0369a1"],
              edgecolor="#0a0d14", width=0.5)
        a.set_ylabel("Count", color="#475569", fontsize=9)
        a.set_title("Property Area", color="#e2e8f0", fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    # Row 2
    st.markdown('<div class="sec-label">Credit & Income Distributions</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        f, a = dark_fig()
        for val, color, lbl in [("Yes","#10b981","Approved"),("No","#ef4444","Rejected")]:
            sub = raw_df[raw_df["Loan_Approved"] == val]["Credit_Score"].dropna()
            a.hist(sub, bins=18, alpha=0.75, color=color, label=lbl, edgecolor="#0a0d14")
        a.axvline(650, color="#fbbf24", linewidth=1.2, linestyle="--", label="650 threshold")
        a.set_xlabel("Credit Score", color="#475569", fontsize=9)
        a.set_title("Credit Score by Outcome", color="#e2e8f0", fontsize=11)
        a.legend(facecolor="#0f1829", labelcolor="white", fontsize=7.5)
        st.pyplot(f, use_container_width=True); plt.close()

    with col5:
        f, a = dark_fig()
        for val, color, lbl in [("Yes","#10b981","Approved"),("No","#ef4444","Rejected")]:
            sub = raw_df[raw_df["Loan_Approved"] == val]["DTI_Ratio"].dropna()
            a.hist(sub, bins=18, alpha=0.75, color=color, label=lbl, edgecolor="#0a0d14")
        a.axvline(0.43, color="#fbbf24", linewidth=1.2, linestyle="--", label="0.43 median")
        a.set_xlabel("DTI Ratio", color="#475569", fontsize=9)
        a.set_title("DTI Ratio by Outcome", color="#e2e8f0", fontsize=11)
        a.legend(facecolor="#0f1829", labelcolor="white", fontsize=7.5)
        st.pyplot(f, use_container_width=True); plt.close()

    with col6:
        f, a = dark_fig()
        a.hist(raw_df["Applicant_Income"].dropna(), bins=25,
               color="#3b82f6", alpha=0.85, edgecolor="#0a0d14")
        a.set_xlabel("Monthly Income (₹)", color="#475569", fontsize=9)
        a.set_title("Applicant Income Distribution", color="#e2e8f0", fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    # Boxplots
    st.markdown('<div class="sec-label">Key Feature Spread by Approval</div>', unsafe_allow_html=True)
    f_box, axes = plt.subplots(1, 3, figsize=(14, 4))
    f_box.patch.set_facecolor("#0f1829")
    feats = ["Credit_Score","DTI_Ratio","Applicant_Income"]
    for ax_b, feat in zip(axes, feats):
        ax_b.set_facecolor("#0f1829")
        data_yes = raw_df[raw_df["Loan_Approved"]=="Yes"][feat].dropna()
        data_no  = raw_df[raw_df["Loan_Approved"]=="No"][feat].dropna()
        bp = ax_b.boxplot([data_no, data_yes],
                          patch_artist=True,
                          labels=["Rejected","Approved"],
                          medianprops={"color":"#fbbf24","linewidth":2})
        bp["boxes"][0].set_facecolor("#7f1d1d")
        bp["boxes"][1].set_facecolor("#064e3b")
        for item in ["whiskers","caps","fliers"]:
            for el in bp[item]: el.set_color("#475569")
        ax_b.set_title(feat.replace("_"," "), color="#e2e8f0", fontsize=10)
        ax_b.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax_b.spines.values(): sp.set_edgecolor("#1e3a5f")
    plt.tight_layout()
    st.pyplot(f_box, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="sec-label">Why Naive Bayes?</div>', unsafe_allow_html=True)

    nb1, nb2, nb3 = st.columns(3)
    nb1.markdown("""
    <div class="input-card">
      <div class="input-card-title">🏆 Highest Precision</div>
      <p style="color:#94a3b8;font-size:0.83rem;line-height:1.6">
        For loan approval, <strong style="color:#f1f5f9">false positives</strong> (approving bad loans)
        cost the bank money. Naive Bayes minimised these, making it the safest business choice.
      </p>
    </div>""", unsafe_allow_html=True)
    nb2.markdown("""
    <div class="input-card">
      <div class="input-card-title">⚡ Fast & Interpretable</div>
      <p style="color:#94a3b8;font-size:0.83rem;line-height:1.6">
        Gaussian Naive Bayes gives <strong style="color:#f1f5f9">probabilistic outputs</strong>
        natively — no calibration needed. It's transparent about how much each feature
        pulls toward approval or rejection.
      </p>
    </div>""", unsafe_allow_html=True)
    nb3.markdown("""
    <div class="input-card">
      <div class="input-card-title">📐 Robust on Small Data</div>
      <p style="color:#94a3b8;font-size:0.83rem;line-height:1.6">
        With 1,000 records and 27 features, NB's independence assumption
        <strong style="color:#f1f5f9">prevents overfitting</strong> that KNN or complex models
        may exhibit.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Performance Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{perf['accuracy']:.2%}")
    m2.metric("Precision", f"{perf['precision']:.2%}", help="Of approved predictions, how many were correct")
    m3.metric("Recall",    f"{perf['recall']:.2%}",    help="Of actual approvals, how many did we catch")
    m4.metric("F1 Score",  f"{perf['f1']:.2%}",        help="Harmonic mean of Precision & Recall")

    st.markdown('<div class="sec-label">Confusion Matrix</div>', unsafe_allow_html=True)

    cm_col, explain_col = st.columns([1, 1], gap="large")

    with cm_col:
        cm = perf["cm"]
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#0f1829")
        ax_cm.set_facecolor("#0f1829")

        cmap = plt.cm.Blues
        im = ax_cm.imshow(cm, cmap=cmap, aspect="auto")

        ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Predicted\nRejected","Predicted\nApproved"], color="#94a3b8", fontsize=9)
        ax_cm.set_yticklabels(["Actual\nRejected","Actual\nApproved"], color="#94a3b8", fontsize=9)

        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=22, fontweight="bold",
                           color="white" if cm[i,j] > cm.max()*0.5 else "#1e3a5f")

        ax_cm.set_title("Naive Bayes — Test Set (200 samples)", color="#94a3b8", fontsize=9, pad=10)
        for sp in ax_cm.spines.values(): sp.set_edgecolor("#1e3a5f")
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True); plt.close()

    with explain_col:
        tn, fp, fn, tp = perf["cm"].ravel()
        st.markdown(f"""
        <div class="input-card" style="margin-top:0.5rem">
          <div class="input-card-title">Reading the Matrix</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#94a3b8">
            <tr>
              <td style="padding:6px 0;border-bottom:1px solid #1e3a5f">
                <strong style="color:#4ade80">True Negatives (TN)</strong>
              </td>
              <td style="text-align:right;padding:6px 0;border-bottom:1px solid #1e3a5f;color:#f1f5f9">{tn}</td>
              <td style="padding:6px 0 6px 12px;border-bottom:1px solid #1e3a5f">Correctly rejected</td>
            </tr>
            <tr>
              <td style="padding:6px 0;border-bottom:1px solid #1e3a5f">
                <strong style="color:#fbbf24">False Positives (FP)</strong>
              </td>
              <td style="text-align:right;padding:6px 0;border-bottom:1px solid #1e3a5f;color:#f1f5f9">{fp}</td>
              <td style="padding:6px 0 6px 12px;border-bottom:1px solid #1e3a5f">Bad loans approved ⚠</td>
            </tr>
            <tr>
              <td style="padding:6px 0;border-bottom:1px solid #1e3a5f">
                <strong style="color:#f97316">False Negatives (FN)</strong>
              </td>
              <td style="text-align:right;padding:6px 0;border-bottom:1px solid #1e3a5f;color:#f1f5f9">{fn}</td>
              <td style="padding:6px 0 6px 12px;border-bottom:1px solid #1e3a5f">Good loans missed</td>
            </tr>
            <tr>
              <td style="padding:6px 0">
                <strong style="color:#4ade80">True Positives (TP)</strong>
              </td>
              <td style="text-align:right;padding:6px 0;color:#f1f5f9">{tp}</td>
              <td style="padding:6px 0 6px 12px">Correctly approved</td>
            </tr>
          </table>
          <p style="margin-top:1rem;font-size:0.78rem;color:#475569;line-height:1.6">
            A Precision of <strong style="color:#f1f5f9">{perf['precision']:.1%}</strong> means
            {fp} out of {tp+fp} approved loans were risky — minimising bank losses.
          </p>
        </div>""", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="sec-label">ML Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("📥", "Raw Data", "1,000 loan records\n19 features"),
        ("🔧", "Imputation", "Mean (numeric)\nMode (categorical)"),
        ("🔠", "Encoding", "LabelEncoder +\nOneHotEncoder"),
        ("⚗️", "Feature Eng.", "DTI² · CreditScore²\nDrop originals"),
        ("⚖️", "Scaling", "StandardScaler\nfit on train only"),
        ("🤖", "Naive Bayes", "GaussianNB\nrandom_state=42"),
        ("📊", "Evaluation", f"Acc {perf['accuracy']:.0%}\nPrec {perf['precision']:.0%}"),
    ]

    cols_pipe = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols_pipe, steps):
        col.markdown(f"""
        <div style="text-align:center;background:#0f1829;border:1px solid #1e3a5f;
                    border-radius:10px;padding:0.9rem 0.5rem">
          <div style="font-size:1.4rem">{icon}</div>
          <div style="font-size:0.78rem;font-weight:600;color:#93c5fd;margin:4px 0">{title}</div>
          <div style="font-size:0.7rem;color:#475569;white-space:pre-line;line-height:1.5">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;border-top:1px solid #1e3a5f;margin-top:3rem">
  <span style="color:#1e3a5f;font-size:0.78rem;letter-spacing:0.06em">
    CREDITWISE · SECURETRUST BANK · BUILT WITH STREAMLIT & SCIKIT-LEARN · NAIVE BAYES ENGINE
  </span>
</div>
""", unsafe_allow_html=True)
