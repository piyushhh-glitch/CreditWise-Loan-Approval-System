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

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — SecureTrust Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# COLOR PALETTE  (from Coolors screenshot)
# ECCBD9 Petal Frost  |  E1EFF6 Alice Blue  |  97D2FB Sky Blue
# 83BCFF Baby Blue Ice  |  80FFE8 Aquamarine
# ─────────────────────────────────────────────────────────────
P = {
    "petal":   "#ECCBD9",
    "alice":   "#E1EFF6",
    "sky":     "#97D2FB",
    "baby":    "#83BCFF",
    "aqua":    "#80FFE8",
    "bg":      "#DAE8F5",          # noticeable blue-grey page bg
    "surface": "#FFFFFF",
    "border":  "#7AAFD4",          # much darker border — visible on white
    "text":    "#0D1F2D",
    "muted":   "#3A5A75",
    "danger":  "#D64045",
    "success": "#1A936F",
    "warn":    "#E8871A",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Lora:wght@400;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: {P['bg']} !important;
    color: {P['text']};
}}

/* ── HERO ── */
.hero {{
    background: linear-gradient(120deg, {P['baby']} 0%, {P['sky']} 40%, {P['aqua']} 100%);
    border-radius: 20px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(131,188,255,0.25);
}}
.hero::after {{
    content: '🏦';
    position: absolute;
    right: 2.5rem; top: 1.5rem;
    font-size: 5rem;
    opacity: 0.12;
    pointer-events: none;
}}
.hero-eyebrow {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #1A4A6E;
    margin-bottom: 0.5rem;
}}
.hero-title {{
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #0D2137;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}}
.hero-sub {{
    color: #1A4A6E;
    font-size: 0.92rem;
    max-width: 500px;
    line-height: 1.65;
    font-weight: 400;
}}
.hero-stats {{
    display: flex;
    gap: 2.2rem;
    margin-top: 1.8rem;
    flex-wrap: wrap;
}}
.hstat {{ display: flex; flex-direction: column; gap: 2px; }}
.hstat-val {{
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #0D2137;
}}
.hstat-lbl {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #2A5F8A;
}}
.nb-badge {{
    display: inline-block;
    background: #0D2137;
    color: {P['aqua']};
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    margin-left: 0.8rem;
    vertical-align: middle;
}}

/* ── DISCLAIMER BANNER ── */
.disclaimer {{
    background: #FFF8F0;
    border: 1px solid #F5C88A;
    border-left: 4px solid {P['warn']};
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    font-size: 0.78rem;
    color: #7A4A00;
    margin-bottom: 1.4rem;
    line-height: 1.55;
}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {P['alice']};
    border: 2px solid #7AAFD4;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    margin-bottom: 1.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {P['muted']};
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.1rem;
}}
.stTabs [aria-selected="true"] {{
    background: {P['baby']} !important;
    color: #0D2137 !important;
    font-weight: 700 !important;
}}

/* ── SECTION LABEL ── */
.sec-lbl {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1A4A6E;
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #5A9CC5;
    display: block;
}}

/* ── CARDS ── */
.card {{
    background: #FFFFFF;
    border: 2px solid #7AAFD4;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(90,156,197,0.15);
}}
.card-title {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #2A5F8A;
    margin-bottom: 0.9rem;
}}

/* ── RESULT PANELS ── */
.result-approved {{
    background: linear-gradient(135deg, #E8F7F2 0%, #C8EFE3 100%);
    border: 2px solid {P['success']};
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(26,147,111,0.15);
}}
.result-rejected {{
    background: linear-gradient(135deg, #FDF0F0 0%, #FAD9D9 100%);
    border: 2px solid {P['danger']};
    border-radius: 16px;
    padding: 2rem 2.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(214,64,69,0.12);
}}
.result-icon {{ font-size: 2.4rem; margin-bottom: 0.4rem; }}
.result-verdict {{
    font-family: 'Lora', serif;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}
.result-sub {{ font-size: 0.88rem; opacity: 0.8; }}

/* ── REASON / IMPROVE CARDS ── */
.reason-card {{
    background: #FDF0F0;
    border: 1px solid #F5BCBC;
    border-left: 4px solid {P['danger']};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
}}
.reason-title {{ font-size: 0.82rem; font-weight: 700; color: #A02020; margin-bottom: 0.25rem; }}
.reason-body  {{ font-size: 0.8rem; color: #5A3535; line-height: 1.6; }}

.improve-card {{
    background: #EEF6FB;
    border: 1px solid {P['sky']};
    border-left: 4px solid {P['baby']};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
}}
.improve-title {{ font-size: 0.82rem; font-weight: 700; color: #1A4A6E; margin-bottom: 0.25rem; }}
.improve-body  {{ font-size: 0.8rem; color: #2A5F8A; line-height: 1.6; }}

/* ── TILE ── */
.tile {{
    background: #FFFFFF;
    border: 2px solid #7AAFD4;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(90,156,197,0.14);
}}
.tile-lbl {{
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: {P['muted']};
}}
.tile-val {{
    font-family: 'Lora', serif;
    font-size: 1.4rem; font-weight: 600;
    color: {P['text']}; margin: 2px 0;
}}
.tile-note {{ font-size: 0.72rem; color: {P['muted']}; }}

/* ── PIPELINE STEP ── */
.pipe-step {{
    background: {P['alice']};
    border: 1px solid {P['border']};
    border-radius: 10px;
    padding: 0.9rem 0.5rem;
    text-align: center;
}}
.pipe-icon  {{ font-size: 1.4rem; }}
.pipe-title {{ font-size: 0.78rem; font-weight: 700; color: #1A4A6E; margin: 4px 0; }}
.pipe-desc  {{ font-size: 0.69rem; color: {P['muted']}; white-space: pre-line; line-height: 1.5; }}

/* ── ABOUT ── */
.about-hero {{
    background: linear-gradient(120deg, {P['petal']} 0%, {P['alice']} 60%, {P['sky']} 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.5rem;
}}
.about-section {{
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(131,188,255,0.07);
}}
.about-section h3 {{
    font-family: 'Lora', serif;
    font-size: 1.05rem;
    color: {P['text']};
    margin-bottom: 0.8rem;
    border-bottom: 2px solid {P['sky']};
    padding-bottom: 0.4rem;
}}
.about-section p, .about-section li {{
    font-size: 0.85rem;
    color: {P['muted']};
    line-height: 1.7;
}}
.about-section ul {{ padding-left: 1.2rem; }}
.metric-pill {{
    display: inline-block;
    background: {P['alice']};
    border: 1px solid {P['sky']};
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #1A4A6E;
    margin: 3px;
}}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stMetric"] {{
    background: #FFFFFF !important;
    border: 2px solid #7AAFD4 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 2px 8px rgba(90,156,197,0.14) !important;
}}
div[data-testid="stMetricLabel"] p {{ color: {P['muted']} !important; font-size: 0.78rem !important; }}
div[data-testid="stMetricValue"]  {{ color: {P['text']}  !important; }}

.stButton > button {{
    background: linear-gradient(135deg, {P['baby']} 0%, {P['sky']} 100%) !important;
    color: #0D2137 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(131,188,255,0.35) !important;
    transition: opacity 0.2s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

label[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stNumberInput label, .stSlider label {{
    color: {P['muted']} !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}}

/* ===== INPUT BOXES ===== */
div[data-baseweb="input"] > div {{
    background: #FFFFFF !important;
    border: 2px solid #5A9CC5 !important;
    border-radius: 8px !important;
}}

div[data-baseweb="input"] input {{
    color: #0D1F2D !important;
    font-weight: 600 !important;
    background: #FFFFFF !important;
}}

div[data-baseweb="select"] > div {{
    background: #FFFFFF !important;
    border: 2px solid #5A9CC5 !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}}

div[data-baseweb="select"] * {{
    color: #0D1F2D !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}}

div[data-baseweb="select"] svg {{
    fill: #0D1F2D !important;
}}

.stSelectbox * {{
    color: #0D1F2D !important;
    opacity: 1 !important;
}}


/* Number input +/- buttons */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {{
    background: {P['sky']} !important;
    border: 1px solid #5A9CC5 !important;
    color: #0D1F2D !important;
}}
/* Slider track */
div[data-testid="stSlider"] div[role="slider"] {{
    background: {P['baby']} !important;
}}
/* Input number value */
input[type="number"] {{
    color: #0D1F2D !important;
    background: #EAF4FB !important;
    font-weight: 600 !important;
}}

.stDataFrame {{ border: 1px solid {P['border']} !important; border-radius: 10px !important; overflow: hidden !important; }}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PIPELINE  (x100 scaled — real rupee values)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training CreditWise engine…")
def build_pipeline():
    df = pd.read_csv("dataset/loan_approval_data.csv")

    # Scale financial columns ×100 so values are real-world INR
    scale_cols = ["Applicant_Income","Coapplicant_Income",
                  "Savings","Collateral_Value","Loan_Amount"]
    for c in scale_cols:
        df[c] = df[c] * 100

    categ_cols = df.select_dtypes(include=["object"]).columns
    num_cols   = df.select_dtypes(include=["float64","int64"]).columns

    df[num_cols]   = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[categ_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[categ_cols])

    raw_df = df.copy()

    df = df.drop(columns=["Applicant_ID"])
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])  # No=0, Yes=1

    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    enc_arr = ohe.fit_transform(df[ohe_cols])
    enc_df  = pd.DataFrame(enc_arr, columns=ohe.get_feature_names_out(), index=df.index)
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

    nb = GaussianNB()
    nb.fit(Xtr_s, y_train)
    yp = nb.predict(Xte_s)

    perf = {
        "accuracy":  accuracy_score(y_test, yp),
        "precision": precision_score(y_test, yp),
        "recall":    recall_score(y_test, yp),
        "f1":        f1_score(y_test, yp),
        "cm":        confusion_matrix(y_test, yp),
    }
    return nb, scaler, ohe, feature_cols, raw_df, df, perf


model, scaler, ohe, feature_cols, raw_df, enc_df, perf = build_pipeline()


# ─────────────────────────────────────────────────────────────
# REJECTION ANALYSER
# ─────────────────────────────────────────────────────────────
def analyse_rejection(credit_score, dti_ratio, applicant_income,
                      coapplicant_income, savings, loan_amount,
                      existing_loans, collateral_value, loan_term,
                      employment_status, age):
    reasons, tips = [], []
    total_inc = applicant_income + coapplicant_income
    lti       = loan_amount / total_inc if total_inc > 0 else 99
    monthly   = loan_amount / loan_term if loan_term > 0 else 0

    if credit_score < 620:
        reasons.append(("Low Credit Score",
            f"Your CIBIL score of {int(credit_score)} is below the preferred threshold of 650. "
            "Lenders treat this as the primary risk indicator."))
        tips.append(("Improve Your Credit Score",
            "Pay all existing EMIs on time for 6+ months. Keep credit card utilisation below 30%. "
            "Dispute any errors in your credit report — even small corrections can lift your score by 20–40 points."))
    elif credit_score < 660:
        reasons.append(("Borderline Credit Score",
            f"Your score of {int(credit_score)} sits in the 620–659 zone. Most approvals happen at 660+."))
        tips.append(("Boost Your Score by 30–40 Points",
            "Avoid any new credit applications for 3–6 months. Clear small overdue balances first — "
            "they have the highest per-rupee impact on your score."))

    if dti_ratio > 0.50:
        reasons.append(("Critical Debt-to-Income Ratio",
            f"A DTI of {dti_ratio:.0%} means more than half your income is already spoken for. "
            "Most lenders cap approvals at 40–50%."))
        tips.append(("Reduce Your Debt Load",
            "Pay off or close one existing loan before reapplying. "
            f"Even reducing DTI by 5–10% can swing the decision."))
    elif dti_ratio > 0.40:
        reasons.append(("Elevated DTI Ratio",
            f"DTI of {dti_ratio:.0%} is above the comfortable 40% ceiling."))
        tips.append(("Lower Your DTI",
            "Consider prepaying the highest-interest loan you hold. "
            "A reduction of ₹2,000–₹3,000/month in obligations can bring DTI under 40%."))

    if lti > 5:
        reasons.append(("Loan-to-Income Ratio Too High",
            f"₹{loan_amount:,.0f} is {lti:.1f}× your combined annual income. "
            "Banks typically approve up to 4–5× annual income for secured loans."))
        tips.append(("Request a Smaller Loan Amount",
            f"Try requesting ₹{total_inc * 4:,.0f} or less. "
            "You can also add a co-applicant with income to raise your eligible amount."))

    if savings < loan_amount * 0.10:
        reasons.append(("Insufficient Savings Buffer",
            f"Savings of ₹{savings:,.0f} are less than 10% of the loan amount. "
            "Lenders want to see a financial cushion for repayment continuity."))
        tips.append(("Build a Savings Buffer",
            f"Accumulate at least ₹{loan_amount*0.12:,.0f} (12% of loan) in a savings or FD account. "
            "This directly signals repayment capacity."))

    if collateral_value < loan_amount * 0.50 and loan_amount > 500000:
        reasons.append(("Low Collateral Coverage",
            f"Collateral of ₹{collateral_value:,.0f} covers only "
            f"{collateral_value/loan_amount:.0%} of the loan. "
            "For large loans, lenders prefer 60–80% coverage."))
        tips.append(("Add or Upgrade Collateral",
            "Property, gold bonds, or a fixed deposit as collateral improves your security ratio significantly."))

    if existing_loans >= 3:
        reasons.append(("Too Many Active Loans",
            f"You currently hold {int(existing_loans)} active loans. "
            "This signals an over-leveraged profile to lenders."))
        tips.append(("Close Existing Loans First",
            "Pay off 1–2 loans before reapplying, starting with the highest-interest ones. "
            "Each closed loan visibly reduces your risk profile."))

    if employment_status == "Unemployed":
        reasons.append(("No Active Income Source",
            "Unemployment means no demonstrated repayment capacity — "
            "a primary requirement for any loan sanction."))
        tips.append(("Secure Employment Before Applying",
            "Wait until you have at least 6 months of stable employment income. "
            "Even part-time or freelance income with bank statements helps."))

    if age < 22:
        reasons.append(("Thin Credit History",
            f"At {int(age)}, you likely have fewer than 2 years of credit history. "
            "Banks prefer applicants with an established repayment track record."))
        tips.append(("Build Credit History First",
            "Start with a secured credit card or a small personal loan repaid on time. "
            "12–18 months of clean repayment history usually satisfies this requirement."))

    if not reasons:
        reasons.append(("Marginal Profile — Combined Risk Factors",
            "No single dominant red flag was found, but the combination of your financial profile "
            "falls just below the model's approval threshold."))
        tips.append(("Small Improvements Across the Board",
            "Improve credit score by 20–30 points, reduce DTI by 5%, "
            "and increase savings by ₹50,000–₹1,00,000 before reapplying."))

    return reasons, tips


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def light_fig(w=5.5, h=4):
    f, a = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor("#FFFFFF")
    a.set_facecolor("#F4F9FD")
    for sp in a.spines.values():
        sp.set_edgecolor(P["border"])
    a.tick_params(colors=P["muted"], labelsize=9)
    return f, a


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
approved_pct = (raw_df["Loan_Approved"] == "Yes").mean() * 100

st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">SecureTrust Bank · AI Credit Division</div>
  <div class="hero-title">CreditWise <span class="nb-badge">Naive Bayes</span></div>
  <div class="hero-sub">Intelligent, unbiased loan assessment powered by machine learning.
    Instant decisions — with full transparency on why.</div>
  <div class="hero-stats">
    <div class="hstat"><span class="hstat-val">{perf['accuracy']:.1%}</span><span class="hstat-lbl">Model Accuracy</span></div>
    <div class="hstat"><span class="hstat-val">{perf['precision']:.1%}</span><span class="hstat-lbl">Precision</span></div>
    <div class="hstat"><span class="hstat-val">{perf['f1']:.1%}</span><span class="hstat-lbl">F1 Score</span></div>
    <div class="hstat"><span class="hstat-val">1,000</span><span class="hstat-lbl">Training Records</span></div>
    <div class="hstat"><span class="hstat-val">{approved_pct:.0f}%</span><span class="hstat-lbl">Historical Approval Rate</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_model, tab_about = st.tabs([
    "🔍  Loan Assessment",
    "📊  Data Insights",
    "🤖  Model Report",
    "ℹ️  About",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════
with tab_predict:

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Dataset note:</strong> This model was trained on a synthetic dataset
      where all monetary values have been scaled ×100 to approximate real-world INR amounts.
      Income range: ₹2L–₹20L/year · Loan range: ₹1L–₹40L · Savings: ₹5K–₹20L.
      Enter values within these ranges for accurate predictions.
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<span class="sec-lbl">Personal Information</span>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            age    = st.number_input("Age", 18, 75, 32)
            gender = st.selectbox("Gender", ["Male","Female"])
        with p2:
            marital_status = st.selectbox("Marital Status", ["Married","Single"])
            dependents     = st.number_input("Dependents", 0, 10, 1)
        with p3:
            education_level = st.selectbox("Education Level", ["Graduate","Not Graduate"])
            property_area   = st.selectbox("Property Area",   ["Urban","Semiurban","Rural"])

        st.markdown('<span class="sec-lbl">Financial Profile</span>', unsafe_allow_html=True)
        f1c, f2c, f3c = st.columns(3)
        with f1c:
            applicant_income    = st.number_input("Applicant Income ₹/yr",    0, 3000000, 800000,  step=10000)
            coapplicant_income  = st.number_input("Co-applicant Income ₹/yr", 0, 2000000, 200000,  step=10000)
        with f2c:
            savings         = st.number_input("Savings (₹)",         0, 2500000, 300000, step=10000)
            collateral_value= st.number_input("Collateral Value (₹)",0, 6000000, 500000, step=25000)
        with f3c:
            existing_loans  = st.number_input("Existing Loans", 0, 10, 1)
            dti_ratio       = st.slider("DTI Ratio", 0.10, 0.60, 0.30, 0.01,
                                        help="Fraction of income already committed to debt")

        st.markdown('<span class="sec-lbl">Credit & Employment</span>', unsafe_allow_html=True)
        e1c, e2c, e3c = st.columns(3)
        with e1c:
            credit_score       = st.slider("Credit Score (CIBIL)", 550, 800, 720)
            employment_status  = st.selectbox("Employment Status",
                ["Salaried","Self-employed","Contract","Unemployed"])
        with e2c:
            employer_category  = st.selectbox("Employer Category",
                ["Private","Government","MNC","Business","Unemployed"])
        with e3c:
            loan_purpose = st.selectbox("Loan Purpose",
                ["Home","Car","Education","Personal","Business"])

        st.markdown('<span class="sec-lbl">Loan Request</span>', unsafe_allow_html=True)
        l1c, l2c = st.columns(2)
        with l1c:
            loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 1500000, step=50000)
        with l2c:
            loan_term   = st.selectbox("Loan Term (months)", [12,24,36,48,60,84,120,180,240,360])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Run Loan Assessment")

    # ── LIVE RISK PANEL ──────────────────────────────────────
    with right_col:
        st.markdown('<span class="sec-lbl">Applicant Risk Dashboard</span>', unsafe_allow_html=True)

        total_income = applicant_income + coapplicant_income
        lti_ratio    = round(loan_amount / total_income, 2) if total_income > 0 else 99
        monthly_emi  = round(loan_amount / loan_term, 0)   if loan_term > 0   else 0
        emi_to_inc   = monthly_emi / (total_income/12)     if total_income > 0 else 1

        if credit_score >= 750:   cs_lbl, cs_col = "Excellent", P["success"]
        elif credit_score >= 700: cs_lbl, cs_col = "Good",      "#2E8B57"
        elif credit_score >= 650: cs_lbl, cs_col = "Fair",      P["warn"]
        elif credit_score >= 600: cs_lbl, cs_col = "Poor",      "#C0681A"
        else:                     cs_lbl, cs_col = "Very Poor", P["danger"]

        if dti_ratio < 0.30:   dti_lbl, dti_col = "Healthy",  P["success"]
        elif dti_ratio < 0.40: dti_lbl, dti_col = "Moderate", P["warn"]
        elif dti_ratio < 0.50: dti_lbl, dti_col = "High",     "#C0681A"
        else:                  dti_lbl, dti_col = "Critical", P["danger"]

        if lti_ratio < 3:      lti_lbl, lti_col = "Conservative", P["success"]
        elif lti_ratio < 5:    lti_lbl, lti_col = "Moderate",     P["warn"]
        else:                  lti_lbl, lti_col = "Aggressive",   P["danger"]

        # Composite risk 0-100
        risk  = max(0, (700 - credit_score) / 1.5)
        risk += min(30, dti_ratio * 50)
        risk += min(20, (lti_ratio - 1) * 4)
        risk += existing_loans * 3
        risk += (0 if employment_status == "Salaried" else 8)
        risk  = min(100, max(0, risk))

        if risk < 30:    rl, rc = "Low Risk",    P["success"]
        elif risk < 55:  rl, rc = "Medium Risk", P["warn"]
        elif risk < 75:  rl, rc = "High Risk",   "#C0681A"
        else:            rl, rc = "Very High",   P["danger"]

        # Gauge
        fig_g, ax_g = plt.subplots(figsize=(4.5, 2.8))
        fig_g.patch.set_facecolor("#FFFFFF")
        ax_g.set_facecolor("#FFFFFF")
        theta = np.linspace(np.pi, 0, 200)
        ax_g.plot(np.cos(theta), np.sin(theta), color=P["border"], linewidth=14, solid_capstyle="round")
        fill_end = np.pi - (risk / 100) * np.pi
        theta_f  = np.linspace(np.pi, fill_end, 200)
        ax_g.plot(np.cos(theta_f), np.sin(theta_f), color=rc, linewidth=14, solid_capstyle="round")
        ax_g.text(0, -0.1, f"{risk:.0f}", ha="center", va="center",
                  fontsize=28, fontweight="bold", color=P["text"])
        ax_g.text(0, -0.42, rl,  ha="center", va="center", fontsize=10, color=rc, fontweight="bold")
        ax_g.text(0, -0.65, "Composite Risk Score", ha="center", va="center",
                  fontsize=7.5, color=P["muted"])
        ax_g.set_xlim(-1.4, 1.4); ax_g.set_ylim(-0.85, 1.1); ax_g.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_g, use_container_width=True); plt.close()

        t1c, t2c = st.columns(2)
        t1c.markdown(f"""<div class="tile"><div class="tile-lbl">Credit Score</div>
            <div class="tile-val" style="color:{cs_col}">{credit_score}</div>
            <div class="tile-note">{cs_lbl}</div></div>""", unsafe_allow_html=True)
        t2c.markdown(f"""<div class="tile"><div class="tile-lbl">DTI Ratio</div>
            <div class="tile-val" style="color:{dti_col}">{dti_ratio:.0%}</div>
            <div class="tile-note">{dti_lbl}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t3c, t4c = st.columns(2)
        t3c.markdown(f"""<div class="tile"><div class="tile-lbl">Loan / Income</div>
            <div class="tile-val" style="color:{lti_col}">{lti_ratio:.1f}×</div>
            <div class="tile-note">{lti_lbl}</div></div>""", unsafe_allow_html=True)
        t4c.markdown(f"""<div class="tile"><div class="tile-lbl">Monthly EMI</div>
            <div class="tile-val">₹{monthly_emi:,.0f}</div>
            <div class="tile-note">{emi_to_inc:.0%} of monthly income</div></div>""",
            unsafe_allow_html=True)

    # ── PREDICTION ────────────────────────────────────────────
    if predict_btn:
        edu_map = {"Graduate": 1, "Not Graduate": 0}
        edu_enc = edu_map.get(education_level, 0)

        ohe_input = pd.DataFrame(
            [[employment_status, marital_status, loan_purpose,
              property_area, gender, employer_category]],
            columns=["Employment_Status","Marital_Status","Loan_Purpose",
                     "Property_Area","Gender","Employer_Category"])
        ohe_enc = pd.DataFrame(ohe.transform(ohe_input), columns=ohe.get_feature_names_out())

        base = pd.DataFrame([{
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age": age, "Dependents": dependents,
            "Existing_Loans": existing_loans,
            "Savings": savings, "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term,
            "Education_Level": edu_enc,
        }])

        full = pd.concat([base.reset_index(drop=True), ohe_enc.reset_index(drop=True)], axis=1)
        full["DTI_Ratio_sq"]    = dti_ratio    ** 2
        full["Credit_Score_sq"] = credit_score ** 2
        for col in feature_cols:
            if col not in full.columns: full[col] = 0
        full   = full[feature_cols]
        scaled = scaler.transform(full)
        pred   = model.predict(scaled)[0]         # 0=No, 1=Yes
        proba  = model.predict_proba(scaled)[0]   # [P(No), P(Yes)]

        st.markdown("---")
        st.markdown('<span class="sec-lbl">Assessment Result</span>', unsafe_allow_html=True)

        rc1, rc2 = st.columns([1, 1], gap="large")
        with rc1:
            if pred == 1:
                st.markdown(f"""<div class="result-approved">
                  <div class="result-icon">✅</div>
                  <div class="result-verdict" style="color:{P['success']}">Loan Approved</div>
                  <div class="result-sub" style="color:#1A5C46">
                    This applicant meets SecureTrust Bank's criteria.<br>
                    Approval confidence: <strong>{proba[1]:.1%}</strong>
                  </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-rejected">
                  <div class="result-icon">❌</div>
                  <div class="result-verdict" style="color:{P['danger']}">Loan Rejected</div>
                  <div class="result-sub" style="color:#7A1A1A">
                    This application does not meet the required criteria.<br>
                    Rejection confidence: <strong>{proba[0]:.1%}</strong>
                  </div></div>""", unsafe_allow_html=True)

        with rc2:
            fig_p, ax_p = plt.subplots(figsize=(4, 2.8))
            fig_p.patch.set_facecolor("#FFFFFF")
            ax_p.set_facecolor("#F4F9FD")
            ax_p.barh(["❌ Rejected","✅ Approved"], [proba[0], proba[1]],
                      color=[P["danger"], P["success"]], height=0.45, edgecolor="white")
            ax_p.set_xlim(0, 1.18)
            ax_p.set_xlabel("Probability", color=P["muted"], fontsize=9)
            ax_p.tick_params(colors=P["muted"], labelsize=9)
            for sp in ax_p.spines.values(): sp.set_edgecolor(P["border"])
            for i, val in enumerate([proba[0], proba[1]]):
                ax_p.text(val + 0.03, i, f"{val:.1%}", va="center",
                          color=P["text"], fontsize=10, fontweight="bold")
            ax_p.set_title("Decision Probability", color=P["muted"], fontsize=9, pad=8)
            plt.tight_layout(); st.pyplot(fig_p, use_container_width=True); plt.close()

        # Rejection breakdown
        if pred == 0:
            reasons, tips = analyse_rejection(
                credit_score, dti_ratio, applicant_income,
                coapplicant_income, savings, loan_amount,
                existing_loans, collateral_value, loan_term,
                employment_status, age)

            st.markdown("<br>", unsafe_allow_html=True)
            rr1, rr2 = st.columns(2, gap="large")
            with rr1:
                st.markdown('<span class="sec-lbl">Why was this rejected?</span>', unsafe_allow_html=True)
                for t, b in reasons:
                    st.markdown(f"""<div class="reason-card">
                      <div class="reason-title">⚠ {t}</div>
                      <div class="reason-body">{b}</div></div>""", unsafe_allow_html=True)
            with rr2:
                st.markdown('<span class="sec-lbl">How to improve eligibility</span>', unsafe_allow_html=True)
                for t, b in tips:
                    st.markdown(f"""<div class="improve-card">
                      <div class="improve-title">💡 {t}</div>
                      <div class="improve-body">{b}</div></div>""", unsafe_allow_html=True)

        # Feature impact
        st.markdown('<span class="sec-lbl">Feature Impact on This Decision</span>', unsafe_allow_html=True)
        scaled_arr = scaled[0]
        log_p1 = -0.5 * ((scaled_arr - model.theta_[1])**2) / model.var_[1]
        log_p0 = -0.5 * ((scaled_arr - model.theta_[0])**2) / model.var_[0]
        delta  = log_p1 - log_p0

        short = {"Applicant_Income":"Applicant Income","Coapplicant_Income":"Co-applicant Income",
                 "Age":"Age","Dependents":"Dependents","Existing_Loans":"Existing Loans",
                 "Savings":"Savings","Collateral_Value":"Collateral","Loan_Amount":"Loan Amount",
                 "Loan_Term":"Loan Term","Education_Level":"Education",
                 "DTI_Ratio_sq":"DTI² (Risk)","Credit_Score_sq":"Credit² (Quality)"}

        dd = pd.DataFrame({"feature": feature_cols, "delta": delta})
        dd["label"] = dd["feature"].map(lambda x: short.get(x, x.replace("_"," ").title()))
        dd = dd.reindex(dd["delta"].abs().nlargest(12).index).sort_values("delta")

        fig_f, ax_f = plt.subplots(figsize=(8, 4))
        fig_f.patch.set_facecolor("#FFFFFF")
        ax_f.set_facecolor("#F4F9FD")
        colors = [P["success"] if v > 0 else P["danger"] for v in dd["delta"]]
        ax_f.barh(dd["label"], dd["delta"], color=colors, edgecolor="white", height=0.6)
        ax_f.axvline(0, color=P["muted"], linewidth=1)
        ax_f.set_xlabel("← Pushes Rejection   |   Pushes Approval →", color=P["muted"], fontsize=8.5)
        ax_f.tick_params(colors=P["muted"], labelsize=9)
        for sp in ax_f.spines.values(): sp.set_edgecolor(P["border"])
        ax_f.set_title("Top 12 Features — Naive Bayes Log-Likelihood Contribution",
                       color=P["muted"], fontsize=9, pad=10)
        plt.tight_layout(); st.pyplot(fig_f, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<span class="sec-lbl">Dataset Overview</span>', unsafe_allow_html=True)
    yes_c = (raw_df["Loan_Approved"] == "Yes").sum()
    no_c  = len(raw_df) - yes_c
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("Total Records",    f"{len(raw_df):,}")
    d2.metric("Approved",         f"{yes_c:,}")
    d3.metric("Rejected",         f"{no_c:,}")
    d4.metric("Approval Rate",    f"{yes_c/len(raw_df):.1%}")
    d5.metric("Features Used",    "27")

    st.markdown('<span class="sec-lbl">Approval & Demographics</span>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)

    with c1:
        f,a = light_fig()
        cnts = raw_df["Loan_Approved"].value_counts()
        a.pie(cnts, labels=["Approved","Rejected"],
              colors=[P["success"], P["danger"]],
              autopct="%1.1f%%",
              textprops={"color": P["text"], "fontsize": 9},
              wedgeprops={"edgecolor":"white","linewidth":2},
              startangle=90)
        a.set_title("Approval Split", color=P["text"], fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    with c2:
        f,a = light_fig()
        emp = raw_df["Employment_Status"].value_counts()
        pal = [P["baby"], P["sky"], P["aqua"], P["petal"], P["alice"]][:len(emp)]
        bars = a.barh(emp.index, emp.values, color=pal, edgecolor="white", height=0.55)
        a.bar_label(bars, padding=4, color=P["text"], fontsize=8)
        a.set_xlabel("Applicants", color=P["muted"], fontsize=9)
        a.set_title("Employment Status", color=P["text"], fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    with c3:
        f,a = light_fig()
        prop = raw_df["Property_Area"].value_counts()
        a.bar(prop.index, prop.values,
              color=[P["sky"], P["baby"], P["aqua"]],
              edgecolor="white", width=0.5)
        a.set_ylabel("Count", color=P["muted"], fontsize=9)
        a.set_title("Property Area", color=P["text"], fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    st.markdown('<span class="sec-lbl">Credit & Income Distributions</span>', unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)

    with c4:
        f,a = light_fig()
        for val, col, lbl in [("Yes", P["success"], "Approved"), ("No", P["danger"], "Rejected")]:
            sub = raw_df[raw_df["Loan_Approved"]==val]["Credit_Score"].dropna()
            a.hist(sub, bins=18, alpha=0.75, color=col, label=lbl, edgecolor="white")
        a.axvline(650, color=P["warn"], linewidth=1.4, linestyle="--", label="650 threshold")
        a.set_xlabel("Credit Score", color=P["muted"], fontsize=9)
        a.set_title("Credit Score by Outcome", color=P["text"], fontsize=11)
        a.legend(fontsize=7.5)
        st.pyplot(f, use_container_width=True); plt.close()

    with c5:
        f,a = light_fig()
        for val, col, lbl in [("Yes", P["success"], "Approved"), ("No", P["danger"], "Rejected")]:
            sub = raw_df[raw_df["Loan_Approved"]==val]["DTI_Ratio"].dropna()
            a.hist(sub, bins=18, alpha=0.75, color=col, label=lbl, edgecolor="white")
        a.axvline(0.43, color=P["warn"], linewidth=1.4, linestyle="--", label="0.43 median")
        a.set_xlabel("DTI Ratio", color=P["muted"], fontsize=9)
        a.set_title("DTI Ratio by Outcome", color=P["text"], fontsize=11)
        a.legend(fontsize=7.5)
        st.pyplot(f, use_container_width=True); plt.close()

    with c6:
        f,a = light_fig()
        a.hist(raw_df["Applicant_Income"].dropna(), bins=25,
               color=P["sky"], alpha=0.85, edgecolor="white")
        a.set_xlabel("Annual Income (₹ — scaled ×100)", color=P["muted"], fontsize=9)
        a.set_title("Income Distribution", color=P["text"], fontsize=11)
        st.pyplot(f, use_container_width=True); plt.close()

    st.markdown('<span class="sec-lbl">Key Features by Approval Outcome</span>', unsafe_allow_html=True)
    f_box, axes = plt.subplots(1, 3, figsize=(14, 4))
    f_box.patch.set_facecolor("#FFFFFF")
    for ax_b, feat in zip(axes, ["Credit_Score","DTI_Ratio","Applicant_Income"]):
        ax_b.set_facecolor("#F4F9FD")
        d_yes = raw_df[raw_df["Loan_Approved"]=="Yes"][feat].dropna()
        d_no  = raw_df[raw_df["Loan_Approved"]=="No"][feat].dropna()
        bp = ax_b.boxplot([d_no, d_yes], patch_artist=True,
                          labels=["Rejected","Approved"],
                          medianprops={"color": P["warn"], "linewidth":2})
        bp["boxes"][0].set_facecolor("#FAD9D9")
        bp["boxes"][1].set_facecolor("#C8EFE3")
        for item in ["whiskers","caps","fliers"]:
            for el in bp[item]: el.set_color(P["muted"])
        ax_b.set_title(feat.replace("_"," "), color=P["text"], fontsize=10)
        ax_b.tick_params(colors=P["muted"], labelsize=9)
        for sp in ax_b.spines.values(): sp.set_edgecolor(P["border"])
    plt.tight_layout()
    st.pyplot(f_box, use_container_width=True); plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL REPORT
# ═══════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<span class="sec-lbl">Why Naive Bayes?</span>', unsafe_allow_html=True)
    nb1,nb2,nb3 = st.columns(3)
    nb1.markdown(f"""<div class="card"><div class="card-title">🏆 Highest Precision</div>
      <p style="color:{P['muted']};font-size:0.83rem;line-height:1.65">
      For loan approval, <strong>false positives</strong> (approving bad loans) cost the bank money.
      Naive Bayes minimised these at <strong>{perf['precision']:.1%}</strong>, making it the safest business choice.
      </p></div>""", unsafe_allow_html=True)
    nb2.markdown(f"""<div class="card"><div class="card-title">⚡ Probabilistic & Fast</div>
      <p style="color:{P['muted']};font-size:0.83rem;line-height:1.65">
      Gaussian NB gives <strong>native probability outputs</strong> — no calibration needed.
      It's fully transparent about how each feature pulls toward approval or rejection.
      </p></div>""", unsafe_allow_html=True)
    nb3.markdown(f"""<div class="card"><div class="card-title">📐 Robust on Small Data</div>
      <p style="color:{P['muted']};font-size:0.83rem;line-height:1.65">
      With 1,000 records and 27 features, NB's independence assumption
      <strong>prevents overfitting</strong> that KNN (66.7% precision) exhibited.
      </p></div>""", unsafe_allow_html=True)

    st.markdown('<span class="sec-lbl">Performance Metrics</span>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{perf['accuracy']:.2%}")
    m2.metric("Precision", f"{perf['precision']:.2%}")
    m3.metric("Recall",    f"{perf['recall']:.2%}")
    m4.metric("F1 Score",  f"{perf['f1']:.2%}")

    st.markdown('<span class="sec-lbl">Confusion Matrix</span>', unsafe_allow_html=True)
    cm_col, ex_col = st.columns([1,1], gap="large")

    with cm_col:
        cm = perf["cm"]
        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
        fig_cm.patch.set_facecolor("#FFFFFF")
        ax_cm.set_facecolor("#F4F9FD")
        ax_cm.imshow(cm, cmap=plt.cm.Blues)
        ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
        ax_cm.set_xticklabels(["Predicted\nRejected","Predicted\nApproved"], color=P["muted"], fontsize=9)
        ax_cm.set_yticklabels(["Actual\nRejected","Actual\nApproved"],      color=P["muted"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                           fontsize=22, fontweight="bold",
                           color="white" if cm[i,j] > cm.max()*0.5 else P["text"])
        ax_cm.set_title("Naive Bayes — Test Set (200 samples)", color=P["muted"], fontsize=9, pad=10)
        for sp in ax_cm.spines.values(): sp.set_edgecolor(P["border"])
        plt.tight_layout(); st.pyplot(fig_cm, use_container_width=True); plt.close()

    with ex_col:
        tn,fp,fn,tp = perf["cm"].ravel()
        st.markdown(f"""<div class="card" style="margin-top:0.5rem">
          <div class="card-title">Reading the Matrix</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <tr><td style="padding:7px 0;border-bottom:1px solid {P['border']}">
              <strong style="color:{P['success']}">True Negatives</strong></td>
              <td style="text-align:right;padding:7px 0;border-bottom:1px solid {P['border']};font-weight:700">{tn}</td>
              <td style="padding:7px 0 7px 12px;border-bottom:1px solid {P['border']};color:{P['muted']}">Correctly rejected</td></tr>
            <tr><td style="padding:7px 0;border-bottom:1px solid {P['border']}">
              <strong style="color:{P['warn']}">False Positives</strong></td>
              <td style="text-align:right;padding:7px 0;border-bottom:1px solid {P['border']};font-weight:700">{fp}</td>
              <td style="padding:7px 0 7px 12px;border-bottom:1px solid {P['border']};color:{P['muted']}">Bad loans approved ⚠</td></tr>
            <tr><td style="padding:7px 0;border-bottom:1px solid {P['border']}">
              <strong style="color:#C0681A">False Negatives</strong></td>
              <td style="text-align:right;padding:7px 0;border-bottom:1px solid {P['border']};font-weight:700">{fn}</td>
              <td style="padding:7px 0 7px 12px;border-bottom:1px solid {P['border']};color:{P['muted']}">Good loans missed</td></tr>
            <tr><td style="padding:7px 0">
              <strong style="color:{P['success']}">True Positives</strong></td>
              <td style="text-align:right;padding:7px 0;font-weight:700">{tp}</td>
              <td style="padding:7px 0 7px 12px;color:{P['muted']}">Correctly approved</td></tr>
          </table>
          <p style="margin-top:0.8rem;font-size:0.78rem;color:{P['muted']};line-height:1.6">
            Precision of <strong>{perf['precision']:.1%}</strong> — only {fp} risky
            loans slipped through out of {tp+fp} approved.
          </p></div>""", unsafe_allow_html=True)

    st.markdown('<span class="sec-lbl">ML Pipeline</span>', unsafe_allow_html=True)
    steps = [("📥","Raw CSV","1,000 records\n19 raw features"),
             ("🔧","Imputation","Mean (numeric)\nMode (categorical)"),
             ("💰","Scale ×100","Financial cols\nreal INR values"),
             ("🔠","Encoding","LabelEncoder +\nOneHotEncoder"),
             ("⚗️","Feature Eng.","DTI² · CreditScore²\nDrop originals"),
             ("⚖️","StandardScaler","Fit on train\nTransform both"),
             ("🤖","GaussianNB","random_state=42\n80/20 split")]
    cols_p = st.columns(len(steps))
    for col, (icon,title,desc) in zip(cols_p, steps):
        col.markdown(f"""<div class="pipe-step">
          <div class="pipe-icon">{icon}</div>
          <div class="pipe-title">{title}</div>
          <div class="pipe-desc">{desc}</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"""
    <div class="about-hero">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;text-transform:uppercase;
                  color:#1A4A6E;margin-bottom:0.5rem">About This Project</div>
      <div style="font-family:'Lora',serif;font-size:2rem;font-weight:600;color:#0D2137;
                  margin-bottom:0.5rem">CreditWise Loan Approval System</div>
      <div style="font-size:0.9rem;color:#1A4A6E;line-height:1.65;max-width:600px">
        An end-to-end supervised machine learning project built to automate and
        de-bias loan approval decisions for SecureTrust Bank — a mid-sized financial
        institution serving urban and rural customers across India.
      </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown(f"""
        <div class="about-section">
          <h3>🎯 Problem Statement</h3>
          <p>SecureTrust Bank processes hundreds of loan applications daily through a
          <strong>manual verification process</strong> that is time-consuming, inconsistent,
          and prone to officer bias. This leads to two costly outcomes:</p>
          <ul>
            <li><strong>Good customers rejected</strong> — loss of business opportunity</li>
            <li><strong>Risky customers approved</strong> — financial losses from defaults</li>
          </ul>
          <p style="margin-top:0.8rem">CreditWise replaces guesswork with a machine learning engine
          that analyses applicant data and predicts approval before final human verification.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section">
          <h3>🗄️ Dataset</h3>
          <p><strong>Source:</strong> Synthetic dataset simulating SecureTrust Bank's historical loan records</p>
          <p><strong>Size:</strong> 1,000 applicant records · 19 raw features · 1 target (Loan_Approved)</p>
          <p style="margin-top:0.6rem"><strong>Key features:</strong></p>
          <ul>
            <li>Applicant & Co-applicant Income, Age, Dependents</li>
            <li>Credit Score (CIBIL), DTI Ratio, Existing Loans</li>
            <li>Savings, Collateral Value, Loan Amount & Term</li>
            <li>Employment Status, Education, Gender, Property Area</li>
          </ul>
          <p style="margin-top:0.8rem;background:#FFF8F0;border-left:3px solid {P['warn']};
             padding:0.6rem 0.8rem;border-radius:6px;font-size:0.79rem;color:#7A4A00">
            ⚠️ <strong>Scaling note:</strong> The original dataset contained monetary values in the
            ₹1K–₹40K range. All financial columns were multiplied by ×100 in preprocessing to
            produce real-world INR amounts (₹1L–₹40L loans, ₹2L–₹20L income) before model training.
            The model accuracy is unchanged — only the input scale changed.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="about-section">
          <h3>🤖 Model Selection</h3>
          <p>Three classifiers were evaluated on the same pipeline:</p>
          <br>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <tr style="background:{P['alice']}">
              <th style="padding:7px 10px;text-align:left;color:{P['text']}">Model</th>
              <th style="padding:7px 10px;text-align:center;color:{P['text']}">Accuracy</th>
              <th style="padding:7px 10px;text-align:center;color:{P['text']}">Precision</th>
              <th style="padding:7px 10px;text-align:center;color:{P['text']}">F1</th>
            </tr>
            <tr style="background:#F0FFF8;border-bottom:1px solid {P['border']}">
              <td style="padding:7px 10px;font-weight:700;color:{P['success']}">✅ Naive Bayes</td>
              <td style="padding:7px 10px;text-align:center">86.5%</td>
              <td style="padding:7px 10px;text-align:center;font-weight:700">78.3%</td>
              <td style="padding:7px 10px;text-align:center">77.7%</td>
            </tr>
            <tr style="background:#FFFFFF;border-bottom:1px solid {P['border']}">
              <td style="padding:7px 10px;color:{P['muted']}">Logistic Regression</td>
              <td style="padding:7px 10px;text-align:center">87.5%</td>
              <td style="padding:7px 10px;text-align:center">79.0%</td>
              <td style="padding:7px 10px;text-align:center">79.7%</td>
            </tr>
            <tr style="background:#FFFFFF">
              <td style="padding:7px 10px;color:{P['muted']}">KNN (k=9)</td>
              <td style="padding:7px 10px;text-align:center">77.0%</td>
              <td style="padding:7px 10px;text-align:center">66.7%</td>
              <td style="padding:7px 10px;text-align:center">56.6%</td>
            </tr>
          </table>
          <p style="margin-top:0.8rem;font-size:0.8rem;color:{P['muted']}">
            Naive Bayes was selected as the production model because it delivers the
            <strong>lowest false positive rate</strong> — approving bad loans is the
            most expensive error for a bank. Its native probability outputs also make
            the rejection explanation system possible.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section">
          <h3>⚙️ Technical Stack</h3>
          <p><span class="metric-pill">Python 3.13</span>
             <span class="metric-pill">Streamlit</span>
             <span class="metric-pill">scikit-learn</span>
             <span class="metric-pill">pandas</span>
             <span class="metric-pill">numpy</span>
             <span class="metric-pill">matplotlib</span>
             <span class="metric-pill">seaborn</span></p>
          <br>
          <p><strong>Pipeline steps:</strong></p>
          <ul>
            <li>Missing value imputation (mean / most-frequent)</li>
            <li>Financial column scaling ×100 (real-world INR)</li>
            <li>Label encoding (Education, Target)</li>
            <li>One-Hot encoding (6 categorical features, drop-first)</li>
            <li>Feature engineering: DTI² and CreditScore²</li>
            <li>StandardScaler (fit on train only)</li>
            <li>Gaussian Naive Bayes, 80/20 split, random_state=42</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-section">
          <h3>👤 Built By</h3>
          <p><strong>Piyush Thakur</strong> — Machine Learning Engineer</p>
          <p style="margin-top:0.3rem">
            <a href="https://github.com/piyushhh-glitch/CreditWise-Loan-Approval-System"
               style="color:#1A4A6E;font-weight:600">
              🔗 GitHub Repository
            </a>
          </p>
          <p style="margin-top:0.3rem;font-size:0.78rem;color:{P['muted']}">
            piyushthakur1611@gmail.com
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2rem 0 0.5rem;border-top:1px solid {P['border']};margin-top:3rem">
  <span style="color:{P['border']};font-size:0.75rem;letter-spacing:0.08em">
    CREDITWISE · SECURETRUST BANK · GAUSSIAN NAIVE BAYES ENGINE · BUILT WITH STREAMLIT
  </span>
</div>
""", unsafe_allow_html=True)
